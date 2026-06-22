use nannou::prelude::*;
use opencv::{core, imgproc, video};

pub struct OpticalFlowResult {
    pub avg_flow: Vec2,
}

pub struct OpticalFlow {
    prev_gray: Option<core::Mat>,
}

impl OpticalFlow {
    pub fn new() -> Self {
        Self { prev_gray: None }
    }

    pub fn get_flow(&mut self, frame: &core::Mat) -> opencv::Result<OpticalFlowResult> {
        let mut gray = core::Mat::default();
        imgproc::cvt_color(
            frame,
            &mut gray,
            imgproc::COLOR_BGR2GRAY,
            0,
            core::AlgorithmHint::ALGO_HINT_DEFAULT,
        )?;

        let mut avg_flow = Vec2::ZERO;
        if let Some(ref prev_gray) = self.prev_gray {
            let mut flow = core::Mat::default();
            video::calc_optical_flow_farneback(
                prev_gray, &gray, &mut flow, 0.5, 3, 15, 3, 5, 1.2, 0,
            )?;

            if let Ok(mean) = core::mean(&flow, &core::no_array()) {
                let dx = mean[0] as f32;
                let dy = -mean[1] as f32;
                avg_flow = vec2(dx, dy);
            }
        }

        self.prev_gray = Some(gray);

        Ok(OpticalFlowResult { avg_flow })
    }
}
