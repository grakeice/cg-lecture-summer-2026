use opencv::{Result, highgui, prelude::*, videoio};

fn main() -> Result<()> {
    let mut cam = videoio::VideoCapture::new(0, videoio::CAP_AVFOUNDATION)?;
    let opened = videoio::VideoCapture::is_opened(&cam)?;
    if !opened {
        panic!("Unable to open default camera (index 0) with AVFoundation backend!");
    }

    let window_name = "OpenCV Camera Preview";
    highgui::named_window(window_name, highgui::WINDOW_AUTOSIZE)?;

    let mut frame = Mat::default();

    println!("Press any key while focusing the camera window to exit.");

    loop {
        cam.read(&mut frame)?;

        if frame.size()?.width > 0 {
            highgui::imshow(window_name, &frame)?;
        }

        let key = highgui::wait_key(30)?;
        if key >= 0 {
            break;
        }
    }

    highgui::destroy_all_windows()?;

    Ok(())
}
