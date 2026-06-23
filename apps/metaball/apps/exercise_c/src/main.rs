use nannou::image::Rgba;
use nannou::prelude::*;

struct Model {
    texture: Handle<Image>,
}

fn main() {
    nannou::app(model).update(update).run();
}

fn model(app: &App) -> Model {
    let _window = app.new_window().size(1000, 1000).view(view).build();

    let assets = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("assets");
    let img_path = assets.join("images/skytree.jpg");

    let image_buffer = nannou::image::open(&img_path)
        .expect("Failed to open image")
        .into_rgba8();

    let width = image_buffer.width();
    let height = image_buffer.height();

    // グレイスケール
    // for x in 0..width {
    //     for y in 0..height {
    //         let mut pixel = *image_buffer.get_pixel(x, y);
    //         let r = pixel[0] as u32;
    //         let g = pixel[1] as u32;
    //         let b = pixel[2] as u32;
    //         let gray = ((r + g + b) / 3) as u8;
    //         pixel.0[0] = gray;
    //         pixel.0[1] = gray;
    //         pixel.0[2] = gray;
    //         image_buffer.put_pixel(x, y, pixel);
    //     }
    // }
    // let dynamic_image = nannou::image::DynamicImage::ImageRgba8(image_buffer);

    // 15x15平均化
    let mut new_image_buffer = nannou::image::RgbaImage::new(width, height);
    let filter_size = 15;
    let half_size = (filter_size / 2) as i32;
    for x in 0..width {
        for y in 0..height {
            let mut r_sum = 0.0;
            let mut g_sum = 0.0;
            let mut b_sum = 0.0;
            for i in -half_size..=half_size {
                for j in -half_size..=half_size {
                    let px = (x as i32 + i).clamp(0, width as i32 - 1) as u32;
                    let py = (y as i32 + j).clamp(0, height as i32 - 1) as u32;
                    r_sum += image_buffer[(px, py)][0] as f32;
                    g_sum += image_buffer[(px, py)][1] as f32;
                    b_sum += image_buffer[(px, py)][2] as f32;
                }
            }
            let filter_area = (filter_size * filter_size) as f32;
            let r = (r_sum / filter_area).clamp(0.0, 255.0) as u8;
            let g = (g_sum / filter_area).clamp(0.0, 255.0) as u8;
            let b = (b_sum / filter_area).clamp(0.0, 255.0) as u8;
            new_image_buffer.put_pixel(x, y, Rgba([r, g, b, 255]));
        }
    }
    let dynamic_image = nannou::image::DynamicImage::ImageRgba8(new_image_buffer);

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
    draw.rect().w_h(1000.0, 1000.0).texture(&model.texture);
}
