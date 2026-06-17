mod ball;

use nannou::prelude::*;

use crate::ball::Ball;

fn main() {
    nannou::app(model).update(update).run();
}

struct Model {
    balls: Vec<ball::Ball>,
}

fn model(app: &App) -> Model {
    let _window = app
        .new_window()
        .size(1024, 1024)
        .view(view)
        .mouse_pressed(mouse_pressed)
        .build()
        .unwrap();

    let balls = vec![];

    let window = app.window(_window).unwrap();
    let device = window.device();

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Metaball"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shaders/metaball.wgsl").into()),
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Metaball pipeline descriptor"),
        bind_group_layouts: &[],
        push_constant_ranges: &[],
    });

    let _render_pipeline = wgpu::RenderPipelineBuilder::from_layout(&pipeline_layout, &shader)
        .primitive_topology(wgpu::PrimitiveTopology::TriangleList)
        .vertex_entry_point("vs_main")
        .fragment_entry_point("fs_main")
        .color_format(Frame::TEXTURE_FORMAT)
        .build(device);

    Model { balls }
}

fn update(_app: &App, _model: &mut Model, _update: Update) {}

fn view(app: &App, model: &Model, frame: Frame) {
    let draw = app.draw();
    draw.background().color(WHITE);

    for ball in &model.balls {
        draw.ellipse()
            .xy(ball.position)
            .radius(ball.radius)
            .color(BLACK);
    }

    draw.to_frame(app, &frame).unwrap();
}

fn mouse_pressed(app: &App, model: &mut Model, button: MouseButton) {
    match button {
        MouseButton::Left => model.balls.push(Ball {
            position: app.mouse.position(),
            radius: 50.0,
        }),
        _ => {}
    }
}
