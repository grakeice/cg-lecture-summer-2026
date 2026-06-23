mod ball;

use nannou::image::{Rgba, RgbaImage};
use nannou::prelude::*;
use std::sync::Arc;

use crate::ball::Ball;

fn main() {
    nannou::app(model).update(update).render(render).run();
}

#[derive(Clone)]
struct Model {
    _window_id: Entity,
    balls: Vec<ball::Ball>,
    image_buffer: RgbaImage,
    texture: Arc<wgpu::Texture>,
    _texture_view: Arc<wgpu::TextureView>,
    pipeline_layout: Arc<wgpu::PipelineLayout>,
    shader: Arc<wgpu::ShaderModule>,
    render_pipeline: Arc<std::sync::OnceLock<wgpu::RenderPipeline>>,
    bind_group: Arc<wgpu::BindGroup>,
}

fn model(app: &App) -> Model {
    let window_id = app
        .new_window()
        .size(1024, 1024)
        .mouse_pressed(mouse_pressed)
        .build();

    let balls = vec![];

    let window = app.window(window_id);
    let device = window.device();

    // Create the offscreen render target texture
    let texture = wgpu::TextureBuilder::new()
        .size([1024, 1024])
        .format(wgpu::TextureFormat::Rgba8Unorm)
        .usage(
            wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST,
        )
        .sample_count(1)
        .build(device);

    let texture_view = texture.view().build();

    let mut image_buffer = RgbaImage::new(1024, 1024);
    for pixel in image_buffer.pixels_mut() {
        *pixel = Rgba([255, 255, 255, 255]);
    }

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Metaball"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shaders/metaball.wgsl").into()),
    });

    let sampler = wgpu::SamplerBuilder::new().build(device);

    let bind_group_layout = wgpu::BindGroupLayoutBuilder::new()
        .texture_from(wgpu::ShaderStages::FRAGMENT, &texture)
        .sampler(wgpu::ShaderStages::FRAGMENT, true)
        .build(device);

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Metaball pipeline descriptor"),
        bind_group_layouts: &[Some(&bind_group_layout)],
        ..Default::default()
    });

    let bind_group = wgpu::BindGroupBuilder::new()
        .texture_view(&texture_view)
        .sampler(&sampler)
        .build(device, &bind_group_layout);

    Model {
        _window_id: window_id,
        balls,
        image_buffer,
        texture: Arc::new(texture),
        _texture_view: Arc::new(texture_view),
        pipeline_layout: Arc::new(pipeline_layout),
        shader: Arc::new(shader),
        render_pipeline: Arc::new(std::sync::OnceLock::new()),
        bind_group: Arc::new(bind_group),
    }
}

fn update(_app: &App, _model: &mut Model) {}

fn render(_app: &RenderApp, model: &Model, frame: Frame) {
    let device = frame.device();

    // 1. Draw original balls display logic using CPU draw to the offscreen texture
    let mut image_buffer = model.image_buffer.clone();

    // 白でクリア
    for pixel in image_buffer.pixels_mut() {
        *pixel = Rgba([255, 255, 255, 255]);
    }

    let width = image_buffer.width() as f32;
    let height = image_buffer.height() as f32;

    for ball in &model.balls {
        let cx = ball.position.x + width / 2.0;
        let cy = height / 2.0 - ball.position.y;
        let r = ball.radius;

        let x_start = ((cx - r).max(0.0) as u32).min(image_buffer.width());
        let x_end = ((cx + r).max(0.0) as u32).min(image_buffer.width());
        let y_start = ((cy - r).max(0.0) as u32).min(image_buffer.height());
        let y_end = ((cy + r).max(0.0) as u32).min(image_buffer.height());

        for x in x_start..x_end {
            for y in y_start..y_end {
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                if dx * dx + dy * dy <= r * r {
                    image_buffer.put_pixel(x, y, Rgba([0, 0, 0, 255]));
                }
            }
        }
    }

    let mut encoder = frame.command_encoder();

    // CPUで描画したピクセルデータをGPUにアップロード
    model
        .texture
        .upload_data(device, &mut encoder, image_buffer.as_raw());

    // 2. Begin the screen pass to render our full-screen triangle post-process shader
    let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("Metaball Render Pass"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: frame.texture_view(),
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                store: StoreOp::Store,
            },
            depth_slice: None,
        })],
        depth_stencil_attachment: None,
        ..Default::default()
    });

    let render_pipeline = model.render_pipeline.get_or_init(|| {
        wgpu::RenderPipelineBuilder::from_layout(&model.pipeline_layout, &model.shader)
            .fragment_shader(&model.shader)
            .primitive_topology(wgpu::PrimitiveTopology::TriangleList)
            .vertex_entry_point("vs_main")
            .fragment_entry_point("fs_main")
            .color_format(frame.texture_format())
            .build(device)
    });

    render_pass.set_pipeline(render_pipeline);
    render_pass.set_bind_group(0, &*model.bind_group, &[]);
    render_pass.draw(0..3, 0..1);
}

fn mouse_pressed(app: &App, model: &mut Model, button: MouseButton) {
    match button {
        MouseButton::Left => model.balls.push(Ball {
            position: app.mouse(),
            radius: 50.0,
        }),
        _ => {}
    }
}
