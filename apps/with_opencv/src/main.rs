mod opencv_utils;
mod face_detector;
mod optical_flow;

use nannou::prelude::*;
use opencv::{core, prelude::*, videoio};
use std::sync::mpsc::{Receiver, channel};
use std::thread;
use std::time::{Duration, Instant};

use opencv_utils::MatExt;

struct Model {
    texture: wgpu::Texture,
    image_receiver: Receiver<nannou::image::RgbaImage>,
    faces_receiver: Receiver<face_detector::FaceDetectorResult>,
    flow_receiver: Receiver<optical_flow::OpticalFlowResult>,
    faces: Vec<core::Rect>,
    avg_flow: Vec2,
}

fn main() {
    nannou::app(model).update(update).run();
}

fn model(app: &App) -> Model {
    let (face_cam_tx, face_cam_rx) = std::sync::mpsc::sync_channel::<core::Mat>(1);
    let (flow_cam_tx, flow_cam_rx) = std::sync::mpsc::sync_channel::<core::Mat>(1);
    let (img_tx, img_rx) = channel::<nannou::image::RgbaImage>();
    let (faces_tx, faces_rx) = channel::<face_detector::FaceDetectorResult>();
    let (flow_tx, flow_rx) = channel::<optical_flow::OpticalFlowResult>();

    // Camera Capture Thread
    thread::spawn(move || {
        let mut cam = videoio::VideoCapture::new(0, videoio::CAP_AVFOUNDATION)
            .expect("Unable to open camera with AVFoundation");

        let opened =
            videoio::VideoCapture::is_opened(&cam).expect("Failed to check if camera is opened");
        if !opened {
            panic!("Camera is not opened!");
        }

        loop {
            let mut raw_frame = Mat::default();
            if let Ok(true) = cam.read(&mut raw_frame) {
                if raw_frame.size().map(|s| s.width > 0).unwrap_or(false) {
                    let mut frame = opencv::core::Mat::default();
                    if core::flip(&raw_frame, &mut frame, 1).is_ok() {
                        let _ = face_cam_tx.try_send(frame.clone());
                        let _ = flow_cam_tx.try_send(frame.clone());

                        if let Ok(rgba_image) = frame.to_rgba_image() {
                            if img_tx.send(rgba_image).is_err() {
                                break;
                            }
                        }
                    }
                }
            }
            thread::sleep(Duration::from_millis(1));
        }
    });

    // Face Detector Thread
    thread::spawn(move || {
        let mut detector = face_detector::FaceDetector::new();
        let process_interval = Duration::from_millis(100);

        loop {
            let start_time = Instant::now();

            let raw_frame = match face_cam_rx.recv() {
                Ok(f) => f,
                Err(_) => break,
            };

            let mut latest_frame = raw_frame;
            while let Ok(f) = face_cam_rx.try_recv() {
                latest_frame = f;
            }

            if let Ok(result) = detector.get_frontalface(&latest_frame) {
                if faces_tx.send(result).is_err() {
                    break;
                }
            }

            let elapsed = start_time.elapsed();
            if elapsed < process_interval {
                thread::sleep(process_interval - elapsed);
            }
        }
    });

    // Optical Flow Thread
    thread::spawn(move || {
        let mut flow_calc = optical_flow::OpticalFlow::new();
        let process_interval = Duration::from_millis(30);

        loop {
            let start_time = Instant::now();

            let raw_frame = match flow_cam_rx.recv() {
                Ok(f) => f,
                Err(_) => break,
            };

            let mut latest_frame = raw_frame;
            while let Ok(f) = flow_cam_rx.try_recv() {
                latest_frame = f;
            }

            if let Ok(result) = flow_calc.get_flow(&latest_frame) {
                if flow_tx.send(result).is_err() {
                    break;
                }
            }

            let elapsed = start_time.elapsed();
            if elapsed < process_interval {
                thread::sleep(process_interval - elapsed);
            }
        }
    });

    let first_image = img_rx.recv().expect("Failed to receive initial frame");
    let width = first_image.width();
    let height = first_image.height();

    let _window = app
        .new_window()
        .size(width, height)
        .view(view)
        .build()
        .unwrap();

    let dynamic_image = nannou::image::DynamicImage::ImageRgba8(first_image);
    let texture = wgpu::Texture::from_image(app, &dynamic_image);

    Model {
        texture,
        image_receiver: img_rx,
        faces_receiver: faces_rx,
        flow_receiver: flow_rx,
        faces: Vec::new(),
        avg_flow: Vec2::ZERO,
    }
}

fn update(app: &App, model: &mut Model, _update: Update) {
    let mut latest_image = None;
    while let Ok(img) = model.image_receiver.try_recv() {
        latest_image = Some(img);
    }

    if let Some(img) = latest_image {
        let window = app.main_window();
        let device = window.device();
        let queue = window.queue();

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Texture Upload"),
        });

        model
            .texture
            .upload_data(device, &mut encoder, img.as_raw());
        queue.submit(Some(encoder.finish()));
    }

    let mut latest_faces = None;
    while let Ok(faces) = model.faces_receiver.try_recv() {
        latest_faces = Some(faces);
    }
    if let Some(res) = latest_faces {
        model.faces = res.faces;
    }

    let mut latest_flow = None;
    while let Ok(flow) = model.flow_receiver.try_recv() {
        latest_flow = Some(flow);
    }
    if let Some(res) = latest_flow {
        model.avg_flow = res.avg_flow;
    }
}

fn view(app: &App, model: &Model, frame: Frame) {
    let draw = app.draw();
    draw.texture(&model.texture);

    let win_rect = app.window_rect();
    let win_width = win_rect.w();
    let win_height = win_rect.h();

    for face in model.faces.iter() {
        let w = face.width as f32;
        let h = face.height as f32;
        let x = (face.x as f32 + w / 2.0) - (win_width / 2.0);
        let y = (win_height / 2.0) - (face.y as f32 + h / 2.0);

        draw.rect()
            .x_y(x, y)
            .w_h(w, h)
            .no_fill()
            .stroke_weight(4.0)
            .stroke_color(STEELBLUE);
    }

    draw.line()
        .start(pt2(0.0, 0.0))
        .end(model.avg_flow * 100.0)
        .color(STEELBLUE)
        .stroke_weight(4.0);

    draw.to_frame(app, &frame).unwrap();
}

