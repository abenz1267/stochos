//! Two Canny thresholds over one shared blur, gradient, and thinning pass.

use image::{GrayImage, Luma};
use imageproc::{
    filter::{filter_clamped, gaussian_blur_f32},
    kernel,
};

pub(super) fn canny_pair(image: &GrayImage, low: f32, high: f32) -> (GrayImage, GrayImage) {
    assert!(high >= low);
    let (width, height) = image.dimensions();
    if width < 3 || height < 3 {
        return (GrayImage::new(width, height), GrayImage::new(width, height));
    }
    let blurred = gaussian_blur_f32(image, 1.4);
    let gx: imageproc::definitions::Image<Luma<i16>> =
        filter_clamped(&blurred, kernel::SOBEL_HORIZONTAL_3X3);
    let gy: imageproc::definitions::Image<Luma<i16>> =
        filter_clamped(&blurred, kernel::SOBEL_VERTICAL_3X3);
    let magnitudes: Vec<f32> = gx
        .iter()
        .zip(gy.iter())
        .map(|(&x, &y)| (x as f32).hypot(y as f32))
        .collect();
    let stride = width as usize;
    let mut thinned = vec![0.0; magnitudes.len()];
    for y in 1..height as usize - 1 {
        for x in 1..stride - 1 {
            let i = y * stride + x;
            let magnitude = magnitudes[i];
            if magnitude == 0.0 {
                continue;
            }
            let dx = gx.as_raw()[i];
            let dy = gy.as_raw()[i];
            // Quantize to the same four directions as imageproc's Canny.
            // Slope comparisons avoid an atan2 for every working-image pixel.
            let ax = dx.abs() as f32;
            let ay = dy.abs() as f32;
            let offset = if ay < ax * 0.414_213_57 {
                1
            } else if ay >= ax * 2.414_213_7 {
                stride
            } else if (dx < 0) == (dy < 0) {
                stride + 1
            } else {
                stride - 1
            };
            if magnitude >= magnitudes[i - offset] && magnitude >= magnitudes[i + offset] {
                thinned[i] = magnitude;
            }
        }
    }
    let mut stack = Vec::new();
    let normal = threshold_edges(&thinned, width, height, low, high, &mut stack);
    let sensitive = threshold_edges(&thinned, width, height, low / 32.0, high / 32.0, &mut stack);
    (normal, sensitive)
}

fn threshold_edges(
    thinned: &[f32],
    width: u32,
    height: u32,
    low: f32,
    high: f32,
    stack: &mut Vec<usize>,
) -> GrayImage {
    let stride = width as usize;
    let mut output = GrayImage::new(width, height);
    let pixels = output.as_mut();
    for y in 1..height as usize - 1 {
        for x in 1..stride - 1 {
            let seed = y * stride + x;
            if thinned[seed] == 0.0 || thinned[seed] < high || pixels[seed] != 0 {
                continue;
            }
            pixels[seed] = 255;
            stack.push(seed);
            while let Some(i) = stack.pop() {
                // Preserve imageproc's hysteresis neighbour stencil so existing
                // outlines and component bounds remain identical.
                for next in [
                    i + 1,
                    i + stride + 1,
                    i + stride,
                    i - stride - 1,
                    i - 1,
                    i + stride - 1,
                ] {
                    // Zero-strength border pixels must stay clear even when a
                    // user configures a zero low threshold.
                    if thinned[next] > 0.0 && thinned[next] >= low && pixels[next] == 0 {
                        pixels[next] = 255;
                        stack.push(next);
                    }
                }
            }
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_pass_matches_independent_canny_passes() {
        let mut state = 42u32;
        for (w, h) in [(3, 3), (17, 31), (128, 96)] {
            let image = GrayImage::from_fn(w, h, |_, _| {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                Luma([(state >> 24) as u8])
            });
            for (low, high) in [(50.0, 150.0), (5.0, 15.0), (100.0, 200.0), (500.0, 1000.0)] {
                let (normal, sensitive) = canny_pair(&image, low, high);
                assert_eq!(normal, imageproc::edges::canny(&image, low, high));
                assert_eq!(
                    sensitive,
                    imageproc::edges::canny(&image, low / 32.0, high / 32.0)
                );
            }
        }
    }

    #[test]
    fn handles_tiny_captures_and_zero_thresholds() {
        for (w, h) in [(1, 1), (1, 8), (8, 1), (2, 2), (8, 8)] {
            let (normal, sensitive) = canny_pair(&GrayImage::new(w, h), 0.0, 0.0);
            assert_eq!(normal.dimensions(), (w, h));
            assert_eq!(sensitive.dimensions(), (w, h));
            assert!(normal.pixels().all(|p| p[0] == 0));
            assert!(sensitive.pixels().all(|p| p[0] == 0));
        }
    }
}
