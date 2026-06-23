mod face_detector;
mod opencv_utils;
mod optical_flow;

use crate::face_detector::{FaceDetector, FaceDetectorResult};
use crate::optical_flow::OpticalFlow;
use nannou::prelude::*;
use opencv::{core, prelude::*, videoio};
use opencv_utils::MatExt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, channel};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

struct Model {
    texture: Handle<Image>,
    image_receiver: Mutex<Receiver<nannou::image::RgbaImage>>,
    faces_receiver: Mutex<Receiver<FaceDetectorResult>>,
    flow_receiver: Mutex<Receiver<core::Mat>>,
    faces: Vec<core::Rect>,
    flow: Option<core::Mat>,
    running: Arc<AtomicBool>,
    thread_handles: Vec<thread::JoinHandle<()>>,
}

fn main() {
    nannou::app(model).update(update).exit(exit).run();
}

fn model(app: &App) -> Model {
    let running = Arc::new(AtomicBool::new(true));
    let mut thread_handles = Vec::new();

    let (face_cam_tx, face_cam_rx) = std::sync::mpsc::sync_channel::<core::Mat>(1);
    let (flow_cam_tx, flow_cam_rx) = std::sync::mpsc::sync_channel::<core::Mat>(1);
    let (img_tx, img_rx) = channel::<nannou::image::RgbaImage>();
    let (faces_tx, faces_rx) = channel::<FaceDetectorResult>();
    let (flow_tx, flow_rx) = channel::<core::Mat>();

    let running_capture = Arc::clone(&running);
    let capture_handle = thread::spawn(move || {
        let mut cam = videoio::VideoCapture::new(0, videoio::CAP_AVFOUNDATION)
            .expect("Unable to open camera with AVFoundation");

        let opened =
            videoio::VideoCapture::is_opened(&cam).expect("Failed to check if camera is opened");
        if !opened {
            panic!("Camera is not opened!");
        }

        while running_capture.load(Ordering::Relaxed) {
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
    thread_handles.push(capture_handle);

    let detector_handle = thread::spawn(move || {
        let mut detector = FaceDetector::new();
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
    thread_handles.push(detector_handle);

    let flow_handle = thread::spawn(move || {
        let mut flow_calc = OpticalFlow::new();
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

            if let Ok(flow) = flow_calc.get_flow(&latest_frame) {
                if flow_tx.send(flow).is_err() {
                    break;
                }
            }

            let elapsed = start_time.elapsed();
            if elapsed < process_interval {
                thread::sleep(process_interval - elapsed);
            }
        }
    });
    thread_handles.push(flow_handle);

    let first_image = img_rx.recv().expect("Failed to receive initial frame");
    let width = first_image.width();
    let height = first_image.height();

    let _window = app.new_window().size(width, height).view(view).build();

    let dynamic_image = nannou::image::DynamicImage::ImageRgba8(first_image);
    let image = Image::from_dynamic(
        dynamic_image,
        true,
        bevy_asset::RenderAssetUsages::default(),
    );
    let texture = app.asset_server().add(image);

    Model {
        texture,
        image_receiver: Mutex::new(img_rx),
        faces_receiver: Mutex::new(faces_rx),
        flow_receiver: Mutex::new(flow_rx),
        faces: Vec::new(),
        flow: None,
        running,
        thread_handles,
    }
}

fn update(app: &App, model: &mut Model) {
    let mut latest_image = None;
    let rx_img = model.image_receiver.get_mut().unwrap();
    while let Ok(img) = rx_img.try_recv() {
        latest_image = Some(img);
    }

    if let Some(img) = latest_image {
        let pixels = img.as_raw().clone();
        app.modify_image(&model.texture, move |image| {
            image.data = Some(pixels);
        });
    }

    let mut latest_faces = None;
    let rx_faces = model.faces_receiver.get_mut().unwrap();
    while let Ok(faces) = rx_faces.try_recv() {
        latest_faces = Some(faces);
    }
    if let Some(res) = latest_faces {
        model.faces = res.faces;
    }

    let mut latest_flow = None;
    let rx_flow = model.flow_receiver.get_mut().unwrap();
    while let Ok(flow) = rx_flow.try_recv() {
        latest_flow = Some(flow);
    }
    if let Some(res) = latest_flow {
        model.flow = Some(res);
    }
}

fn view(app: &App, model: &Model) {
    let draw = app.draw();

    let win_rect = app.window_rect();
    let win_width = win_rect.w();
    let win_height = win_rect.h();

    draw.rect()
        .w_h(win_width, win_height)
        .texture(&model.texture);

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
            .stroke_color(STEEL_BLUE);
    }

    if let Some(flow) = &model.flow {
        if let Ok(avg_flow) = OpticalFlow::get_average_flow(flow) {
            if avg_flow.length_squared() > 1e-6 {
                draw.line()
                    .start(pt2(0.0, 0.0))
                    .end(avg_flow * 100.0)
                    .color(STEEL_BLUE)
                    .stroke_weight(4.0);
            }
        }

        for face in model.faces.iter() {
            if let Ok(face_flow) = OpticalFlow::get_average_flow_in_region(
                flow,
                vec2(face.x as f32, face.y as f32),
                vec2(face.width as f32, face.height as f32),
            ) {
                let w = face.width as f32;
                let h = face.height as f32;
                let x = (face.x as f32 + w / 2.0) - (win_width / 2.0);
                let y = (win_height / 2.0) - (face.y as f32 + h / 2.0);

                if face_flow.length_squared() > 1e-6 {
                    draw.line()
                        .start(pt2(x, y))
                        .end(pt2(x + face_flow.x * 100.0, y + face_flow.y * 100.0))
                        .color(RED)
                        .stroke_weight(4.0);
                }
            }
        }
    }
}

fn exit(_app: &App, model: Model) {
    model.running.store(false, Ordering::Relaxed);

    // Drop receivers explicitly to wake up and exit threads blocked on sends
    drop(model.image_receiver);
    drop(model.faces_receiver);
    drop(model.flow_receiver);

    for handle in model.thread_handles {
        let _ = handle.join();
    }
}
