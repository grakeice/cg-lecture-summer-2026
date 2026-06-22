use nannou::prelude::*;
use opencv::{core, imgproc, prelude::*, video};

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

    pub fn get_flow(&mut self, frame: &core::Mat) -> opencv::Result<core::Mat> {
        let mut gray = core::Mat::default();
        imgproc::cvt_color(
            frame,
            &mut gray,
            imgproc::COLOR_BGR2GRAY,
            0,
            core::AlgorithmHint::ALGO_HINT_DEFAULT,
        )?;

        let mut flow = core::Mat::default();
        if let Some(ref prev_gray) = self.prev_gray {
            video::calc_optical_flow_farneback(
                prev_gray, &gray, &mut flow, 0.5, 3, 15, 3, 5, 1.2, 0,
            )?;
        }

        self.prev_gray = Some(gray);

        Ok(flow)
    }

    pub fn get_average_flow_in_region(
        flow: &core::Mat,
        xy: Vec2,
        wh: Vec2,
    ) -> opencv::Result<Vec2> {
        let mut avg_flow = Vec2::ZERO;
        let region = core::Rect::new(xy.x as i32, xy.y as i32, wh.x as i32, wh.y as i32);
        let roi = flow.roi(region)?;
        if let Ok(mean) = core::mean(&roi, &core::no_array()) {
            let dx = mean[0] as f32;
            let dy = -mean[1] as f32;
            avg_flow = vec2(dx, dy);
        }

        Ok(avg_flow)
    }

    pub fn get_average_flow(&mut self, frame: &core::Mat) -> opencv::Result<OpticalFlowResult> {
        let flow = self.get_flow(frame)?;
        let avg_flow = OpticalFlow::get_average_flow_in_region(
            &flow,
            vec2(0.0, 0.0),
            vec2(frame.cols() as f32, frame.rows() as f32),
        )?;

        Ok(OpticalFlowResult { avg_flow })
    }

    pub fn _get_flow_at(flow: &core::Mat, xy: Vec2) -> opencv::Result<Vec2> {
        let mut avg_flow = Vec2::ZERO;
        let region = core::Rect::new(xy.x as i32, xy.y as i32, 1, 1);
        let roi = flow.roi(region)?;
        if let Ok(mean) = core::mean(&roi, &core::no_array()) {
            let dx = mean[0] as f32;
            let dy = -mean[1] as f32;
            avg_flow = vec2(dx, dy);
        }

        Ok(avg_flow)
    }
}
