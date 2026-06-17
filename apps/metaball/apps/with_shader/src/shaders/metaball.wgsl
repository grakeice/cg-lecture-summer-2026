struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

struct Ball {
    position: vec2<f32>,
    radius: f32,
    _pad: f32,
}

struct MetaballData {
    num_balls: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    balls: array<Ball, 256>,
}

@group(0) @binding(0)
var<uniform> data: MetaballData;

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    let uv = vec2<f32>(
        f32((vertex_index << 1u) & 2u),
        f32(vertex_index & 2u)
    );
    out.uv = uv;
    out.position = vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
    return out;
}

fn quantize_n_levels(value: f32, n: f32) -> f32 {
    if (n < 2.0) {
        return 0.0;
    }
    let step = 1.0 / (n - 1.0);
    let index = round(value / step);
    return index * step;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let coord = (in.uv - vec2<f32>(0.5, 0.5)) * 1024.0;

    var sum: f32 = 0.0;
    for (var i: u32 = 0u; i < data.num_balls; i = i + 1u) {
        let ball = data.balls[i];
        let d = distance(coord, ball.position);
        if (d > 0.0) {
            sum = sum + (50.0 * ball.radius) / d;
        }
    }

    let normalized_sum = clamp(sum / 255.0, 0.0, 1.0);
    let value = quantize_n_levels(normalized_sum, 4.0);

    return vec4<f32>(value, value, value, 1.0);
}