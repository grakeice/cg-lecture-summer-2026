use nannou::image::{Rgba, RgbaImage};
use nannou::prelude::*;

struct Model {
    texture: Handle<Image>,
    image_buffer: RgbaImage,
}

fn main() {
    nannou::app(model).update(update).run();
}

fn model(app: &App) -> Model {
    let _window = app.new_window().size(512, 512).view(view).build();

    let width = 512;
    let height = 512;
    let image_buffer = RgbaImage::new(width, height);

    let dynamic_image = nannou::image::DynamicImage::ImageRgba8(image_buffer.clone());
    let image = Image::from_dynamic(
        dynamic_image,
        true,
        bevy_asset::RenderAssetUsages::default(),
    );
    let texture = app.asset_server().add(image);

    Model {
        texture,
        image_buffer,
    }
}

fn update(app: &App, model: &mut Model) {
    let width = model.image_buffer.width();
    let height = model.image_buffer.height();

    for x in 0..width {
        for y in 0..height {
            let pixel = pt2(x as f32, y as f32);
            let distance = (pixel - pt2(width as f32, height as f32) / 2.0).length();
            let value = if distance < 100.0 {
                (1.0 - distance / 100.0).powf(2.0) * 255.0
            } else {
                0.0
            };
            let value_u8 = value as u8;
            model
                .image_buffer
                .put_pixel(x, y, Rgba([value_u8, value_u8, value_u8, 255]));
        }
    }

    let pixels = model.image_buffer.as_raw().clone();
    app.modify_image(&model.texture, move |image| {
        image.data = Some(pixels);
    });
}

fn view(app: &App, model: &Model) {
    let draw = app.draw();
    draw.rect().w_h(512.0, 512.0).texture(&model.texture);
}
