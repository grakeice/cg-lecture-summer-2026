use opencv::{core, imgproc, objdetect, prelude::*};

pub struct FaceDetectorResult {
    pub faces: Vec<core::Rect>,
}

pub struct FaceDetector {
    face_cascade: objdetect::CascadeClassifier,
}

impl FaceDetector {
    pub fn new() -> Self {
        let manifest_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let face_cascade_path =
            manifest_dir.join("assets/haarcascades/haarcascade_frontalface_default.xml");
        let face_cascade_path_str = face_cascade_path.to_str().unwrap().to_string();

        let face_cascade = objdetect::CascadeClassifier::new(&face_cascade_path_str)
            .expect("Failed to load face cascade");

        Self { face_cascade }
    }

    pub fn get_frontalface(&mut self, frame: &core::Mat) -> opencv::Result<FaceDetectorResult> {
        let mut gray = core::Mat::default();
        imgproc::cvt_color(
            frame,
            &mut gray,
            imgproc::COLOR_BGR2GRAY,
            0,
            core::AlgorithmHint::ALGO_HINT_DEFAULT,
        )?;

        let mut faces = core::Vector::<core::Rect>::new();
        self.face_cascade.detect_multi_scale(
            &gray,
            &mut faces,
            1.1,
            3,
            0,
            core::Size::new(30, 30),
            core::Size::new(0, 0),
        )?;

        Ok(FaceDetectorResult {
            faces: faces.to_vec(),
        })
    }
}
