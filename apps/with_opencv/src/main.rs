mod face_detector;
mod opencv_utils;
mod optical_flow;

use crate::face_detector::{FaceDetector, FaceDetectorResult};
use crate::optical_flow::OpticalFlow;
use bevy::prelude::{
    App as BevyApp, Assets, Handle, Plugin as BevyPlugin, Query, Res, Resource, Update,
};
use nannou::prelude::*;
use nannou_webcam::{Webcam, WebcamPlugin, WebcamStream};
use opencv::core;
use opencv_utils::ImageExt;
use std::sync::Mutex;
use std::sync::mpsc::{Receiver, channel};
use std::thread;
use std::time::{Duration, Instant};

#[derive(Resource)]
struct WebcamChannels {
    face_tx: std::sync::mpsc::SyncSender<core::Mat>,
    flow_tx: std::sync::mpsc::SyncSender<core::Mat>,
    stream_tx: std::sync::mpsc::Sender<(Handle<Image>, UVec2)>,
}

struct WebcamProcessingPlugin;

impl BevyPlugin for WebcamProcessingPlugin {
    fn build(&self, app: &mut BevyApp) {
        app.add_systems(Update, webcam_processing_system);
    }
}

fn webcam_processing_system(
    query: Query<&WebcamStream>,
    images: Res<Assets<Image>>,
    channels: Res<WebcamChannels>,
) {
    for stream in query.iter() {
        let _ = channels
            .stream_tx
            .send((stream.image.clone(), stream.resolution));

        if let Some(image) = images.get(&stream.image) {
            if let Ok(mat) = image.to_mat() {
                let mut flipped = opencv::core::Mat::default();
                if opencv::core::flip(&mat, &mut flipped, 1).is_ok() {
                    let _ = channels.face_tx.try_send(flipped.clone());
                    let _ = channels.flow_tx.try_send(flipped);
                }
            }
        }
    }
}

struct Model {
    texture: Option<Handle<Image>>,
    faces_receiver: Mutex<Receiver<FaceDetectorResult>>,
    flow_receiver: Mutex<Receiver<core::Mat>>,
    faces: Vec<core::Rect>,
    flow: Option<core::Mat>,
    thread_handles: Vec<thread::JoinHandle<()>>,
    cam_size: Vec2,
    webcam_stream_rx: Mutex<Receiver<(Handle<Image>, UVec2)>>,
}

fn main() {
    nannou::app(model)
        .add_plugin(WebcamPlugin)
        .add_plugin(WebcamProcessingPlugin)
        .update(update)
        .exit(exit)
        .run();
}

fn model(app: &App) -> Model {
    let mut thread_handles = Vec::new();

    let (face_cam_tx, face_cam_rx) = std::sync::mpsc::sync_channel::<core::Mat>(1);
    let (flow_cam_tx, flow_cam_rx) = std::sync::mpsc::sync_channel::<core::Mat>(1);
    let (webcam_stream_tx, webcam_stream_rx) = std::sync::mpsc::channel::<(Handle<Image>, UVec2)>();
    let (faces_tx, faces_rx) = channel::<FaceDetectorResult>();
    let (flow_tx, flow_rx) = channel::<core::Mat>();

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

    app.command_scope(|mut commands| {
        commands.insert_resource(WebcamChannels {
            face_tx: face_cam_tx,
            flow_tx: flow_cam_tx,
            stream_tx: webcam_stream_tx,
        });

        commands.spawn(Webcam::default());
    });

    let win_w = 1280;
    let win_h = 720;

    let _window = app.new_window().size(win_w, win_h).view(view).build();

    Model {
        texture: None,
        faces_receiver: Mutex::new(faces_rx),
        flow_receiver: Mutex::new(flow_rx),
        faces: Vec::new(),
        flow: None,
        thread_handles,
        cam_size: vec2(win_w as f32, win_h as f32),
        webcam_stream_rx: Mutex::new(webcam_stream_rx),
    }
}

fn update(_app: &App, model: &mut Model) {
    let rx = model.webcam_stream_rx.get_mut().unwrap();
    while let Ok((handle, resolution)) = rx.try_recv() {
        model.texture = Some(handle);
        model.cam_size = vec2(resolution.x as f32, resolution.y as f32);
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
    draw.background().color(BLACK);

    if let Some(texture) = &model.texture {
        let win_rect = app.window_rect();
        let win_width = win_rect.w();
        let win_height = win_rect.h();

        let cam_aspect = model.cam_size.x / model.cam_size.y;
        let win_aspect = win_width / win_height;

        let (draw_w, draw_h) = if win_aspect > cam_aspect {
            (win_height * cam_aspect, win_height)
        } else {
            (win_width, win_width / cam_aspect)
        };

        draw.scale_x(-1.0)
            .rect()
            .w_h(draw_w, draw_h)
            .texture(texture);

        let scale_x = draw_w / model.cam_size.x;
        let scale_y = draw_h / model.cam_size.y;

        for face in model.faces.iter() {
            let w = face.width as f32 * scale_x;
            let h = face.height as f32 * scale_y;

            let face_center_x = face.x as f32 + face.width as f32 / 2.0;
            let face_center_y = face.y as f32 + face.height as f32 / 2.0;

            let x = (face_center_x - model.cam_size.x / 2.0) * scale_x;
            let y = (model.cam_size.y / 2.0 - face_center_y) * scale_y;

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
                    let face_center_x = face.x as f32 + face.width as f32 / 2.0;
                    let face_center_y = face.y as f32 + face.height as f32 / 2.0;
                    let x = (face_center_x - model.cam_size.x / 2.0) * scale_x;
                    let y = (model.cam_size.y / 2.0 - face_center_y) * scale_y;

                    if face_flow.length_squared() > 1e-6 {
                        let flow_scale = 100.0 * scale_x;
                        draw.line()
                            .start(pt2(x, y))
                            .end(pt2(
                                x + face_flow.x * flow_scale,
                                y + face_flow.y * flow_scale,
                            ))
                            .color(RED)
                            .stroke_weight(4.0);
                    }
                }
            }
        }
    }
}

fn exit(_app: &App, model: Model) {
    drop(model.faces_receiver);
    drop(model.flow_receiver);
    drop(model.webcam_stream_rx);

    for handle in model.thread_handles {
        let _ = handle.join();
    }
}
