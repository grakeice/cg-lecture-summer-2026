use nannou::image::{Rgba, RgbaImage};
use nannou::prelude::*;

struct Model {
    texture: Handle<Image>,
}

fn main() {
    nannou::app(model).update(update).run();
}

fn model(app: &App) -> Model {
    let _window = app.new_window().size(512, 512).view(view).build();

    let width = 512;
    let height = 512;
    let mut image_buffer = RgbaImage::new(width, height);

    // // random
    // for x in 0..width {
    //     for y in 0..height {
    //         let value = random_range(0, 255);
    //         image_buffer.put_pixel(x, y, Rgba([value, value, value, 255]));
    //     }
    // }

    // stripe
    for x in 0..width {
        for y in 0..height {
            if x % 3 == 0 {
                image_buffer.put_pixel(x, y, Rgba([255, 255, 255, 255]));
            } else {
                image_buffer.put_pixel(x, y, Rgba([0, 0, 0, 255]));
            }
        }
    }

    for x in 0..width {
        for y in 0..height {
            let mut pixel = *image_buffer.get_pixel(x, y);
            if 100 < x && x <= 300 && 100 < y && y <= 300 {
                pixel.0[2] = pixel.0[2].saturating_add(100);
                image_buffer.put_pixel(x, y, pixel);
            }
        }
    }

    let dynamic_image = nannou::image::DynamicImage::ImageRgba8(image_buffer);
    let image = Image::from_dynamic(
        dynamic_image,
        true,
        bevy_asset::RenderAssetUsages::default(),
    );
    let texture = app.asset_server().add(image);

    Model { texture }
}

fn update(_app: &App, _model: &mut Model) {}

fn view(app: &App, model: &Model) {
    let draw = app.draw();
    draw.rect().w_h(512.0, 512.0).texture(&model.texture);
}
