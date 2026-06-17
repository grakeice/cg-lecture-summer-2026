mod metaball;

use nannou::image::{Rgba, RgbaImage};
use nannou::prelude::*;

use crate::metaball::Metaball;

struct Model {
    texture: wgpu::Texture,
    metaballs: Vec<Metaball>,
}

fn main() {
    nannou::app(model).update(update).run();
}

fn model(app: &App) -> Model {
    let _window = app
        .new_window()
        .size(1024, 1024)
        .view(view)
        .build()
        .unwrap();

    let width = 1024;
    let height = 1024;

    let metaballs = vec![
        Metaball {
            position: pt2(-50.0, 0.0),
            radius: 100.0,
        },
        Metaball {
            position: pt2(50.0, 0.0),
            radius: 100.0,
        },
    ];

    let mut image_buffer = RgbaImage::new(width, height);
    let dynamic_image = nannou::image::DynamicImage::ImageRgba8(image_buffer.clone());
    let texture = wgpu::Texture::from_image(app, &dynamic_image);

    for x in 0..width {
        for y in 0..height {
            let px = x as f32 - width as f32 / 2.0;
            let py = height as f32 / 2.0 - y as f32;
            let pixel = pt2(px, py);

            let mut sum = 0.0;
            for metaball in &metaballs {
                sum += 50.0 * metaball.radius / (pixel - metaball.position).length();
            }
            let value = quantize_n_levels(sum as u8, 4);
            image_buffer.put_pixel(x, y, Rgba([value, value, value, 255]));
        }
    }

    let window = app.main_window();
    let device = window.device();
    let queue = window.queue();
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Texture Upload"),
    });

    texture.upload_data(device, &mut encoder, image_buffer.as_raw());
    queue.submit(Some(encoder.finish()));

    Model { texture, metaballs }
}

fn update(_app: &App, _model: &mut Model, _update: Update) {}

fn view(app: &App, model: &Model, frame: Frame) {
    let draw = app.draw();
    draw.background().color(BLACK);

    draw.texture(&model.texture);

    // for metaball in model.metaballs.iter() {
    //     draw.ellipse()
    //         .xy(metaball.position)
    //         .radius(metaball.radius)
    //         .color(BLACK);
    // }

    draw.to_frame(app, &frame).unwrap();
}

fn quantize_n_levels(value: u8, n: u8) -> u8 {
    if n < 2 {
        return 0;
    }
    if n >= 255 {
        return value;
    }
    let step = 255.0 / (n - 1) as f32;
    let index = (value as f32 / step).round() as u8;
    (index as f32 * step).round() as u8
}
