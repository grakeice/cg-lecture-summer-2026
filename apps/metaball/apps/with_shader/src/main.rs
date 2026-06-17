mod ball;

use nannou::prelude::*;

use crate::ball::Ball;

fn main() {
    nannou::app(model).update(update).run();
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct BallGPU {
    position: [f32; 2],
    radius: f32,
    _pad: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct MetaballData {
    num_balls: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    balls: [BallGPU; 256],
}

struct Model {
    window_id: window::Id,
    balls: Vec<ball::Ball>,
    render_pipeline: wgpu::RenderPipeline,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

fn model(app: &App) -> Model {
    let window_id = app
        .new_window()
        .size(1024, 1024)
        .view(view)
        .mouse_pressed(mouse_pressed)
        .msaa_samples(1)
        .build()
        .unwrap();

    let balls = vec![];

    let window = app.window(window_id).unwrap();
    let device = window.device();

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Metaball"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shaders/metaball.wgsl").into()),
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Metaball Bind Group Layout"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Metaball pipeline descriptor"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let render_pipeline = wgpu::RenderPipelineBuilder::from_layout(&pipeline_layout, &shader)
        .fragment_shader(&shader)
        .primitive_topology(wgpu::PrimitiveTopology::TriangleList)
        .vertex_entry_point("vs_main")
        .fragment_entry_point("fs_main")
        .color_format(Frame::TEXTURE_FORMAT)
        .build(device);

    let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Metaball Uniform Buffer"),
        size: std::mem::size_of::<MetaballData>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Metaball Bind Group"),
        layout: &bind_group_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: uniform_buffer.as_entire_binding(),
        }],
    });

    Model {
        window_id,
        balls,
        render_pipeline,
        uniform_buffer,
        bind_group,
    }
}

fn update(_app: &App, _model: &mut Model, _update: Update) {}

fn view(app: &App, model: &Model, frame: Frame) {
    let window = app.window(model.window_id).unwrap();
    let queue = window.queue();

    // Prepare metaball data for the shader
    let mut balls_gpu = [BallGPU {
        position: [0.0, 0.0],
        radius: 0.0,
        _pad: 0.0,
    }; 256];
    let num_balls = model.balls.len().min(256);
    for i in 0..num_balls {
        balls_gpu[i] = BallGPU {
            position: [model.balls[i].position.x, model.balls[i].position.y],
            radius: model.balls[i].radius,
            _pad: 0.0,
        };
    }

    let metaball_data = MetaballData {
        num_balls: num_balls as u32,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
        balls: balls_gpu,
    };

    queue.write_buffer(&model.uniform_buffer, 0, bytemuck::bytes_of(&metaball_data));

    let mut encoder = frame.command_encoder();
    let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("Metaball Render Pass"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: frame.texture_view(),
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                store: true,
            },
        })],
        depth_stencil_attachment: None,
    });

    render_pass.set_pipeline(&model.render_pipeline);
    render_pass.set_bind_group(0, &model.bind_group, &[]);
    render_pass.draw(0..3, 0..1);
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

