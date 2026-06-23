use bevy::prelude::Image;
use opencv::core::{Mat, Vec4b};
use opencv::prelude::*;

pub trait ImageExt {
    fn to_mat(&self) -> opencv::Result<Mat>;
}

impl ImageExt for Image {
    fn to_mat(&self) -> opencv::Result<Mat> {
        let width = self.texture_descriptor.size.width as i32;
        let height = self.texture_descriptor.size.height as i32;
        let data = self
            .data
            .as_ref()
            .ok_or_else(|| opencv::Error::new(opencv::core::StsNullPtr, "Image data is None"))?;

        // Bevy's Image contains RGBA pixel data
        let rgba_mat = Mat::new_rows_cols_with_bytes::<Vec4b>(height, width, data)?;
        let owned_rgba = rgba_mat.try_clone()?;

        let mut bgr_mat = Mat::default();
        opencv::imgproc::cvt_color(
            &owned_rgba,
            &mut bgr_mat,
            opencv::imgproc::COLOR_RGBA2BGR,
            0,
            opencv::core::AlgorithmHint::ALGO_HINT_DEFAULT,
        )?;

        Ok(bgr_mat)
    }
}
