//! Binary morphology and region bounds without full pixel-label images.

use super::Component;
use image::GrayImage;
use imageproc::region_labelling::Connectivity;

pub(super) fn close_binary(image: &GrayImage, radius: usize) -> GrayImage {
    let (w, h) = (image.width() as usize, image.height() as usize);
    if w <= radius || h <= radius {
        return imageproc::morphology::close(
            image,
            imageproc::distance_transform::Norm::LInf,
            radius as u8,
        );
    }
    let mut a = image.as_raw().clone();
    let mut b = vec![0; a.len()];
    for dilate in [true, false] {
        // Separable square max/min filters. Pixels outside the image are
        // ignored, matching imageproc's L-infinity distance transform.
        for y in 0..h {
            for x in 0..w {
                let mut value = if dilate { 0 } else { 255 };
                for &p in &a[y * w + x.saturating_sub(radius)..y * w + (x + radius + 1).min(w)] {
                    value = if dilate { value | p } else { value & p };
                }
                b[y * w + x] = value;
            }
        }
        for y in 0..h {
            for x in 0..w {
                let mut value = if dilate { 0 } else { 255 };
                for yy in y.saturating_sub(radius)..(y + radius + 1).min(h) {
                    value = if dilate {
                        value | b[yy * w + x]
                    } else {
                        value & b[yy * w + x]
                    };
                }
                a[y * w + x] = value;
            }
        }
    }
    GrayImage::from_raw(image.width(), image.height(), a).unwrap()
}

struct Run {
    start: u32,
    end: u32,
    label: usize,
}

fn root(parents: &mut [usize], mut i: usize) -> usize {
    while parents[i] != i {
        parents[i] = parents[parents[i]];
        i = parents[i];
    }
    i
}

pub(super) fn component_boxes(
    image: &GrayImage,
    connectivity: Connectivity,
    inverse: bool,
) -> Vec<Component> {
    let w = image.width();
    let diagonal = u32::from(connectivity == Connectivity::Eight);
    let mut previous: Vec<Run> = Vec::new();
    let mut current = Vec::new();
    let mut parents = Vec::new();
    let mut runs = Vec::new();
    for (y, row) in image.as_raw().chunks(w.max(1) as usize).enumerate() {
        let mut x = 0;
        let mut previous_index = 0;
        while x < w {
            if (row[x as usize] == 0) != inverse {
                x += 1;
                continue;
            }
            let start = x;
            while x < w && (row[x as usize] == 0) == inverse {
                x += 1;
            }
            let label = parents.len();
            parents.push(label);
            runs.push(Component {
                bbox: (start, y as u32, x - start, 1),
                pixels: x - start,
            });
            while previous_index < previous.len()
                && previous[previous_index].end + diagonal <= start
            {
                previous_index += 1;
            }
            for prior in &previous[previous_index..] {
                if prior.start >= x + diagonal {
                    break;
                }
                let a = root(&mut parents, label);
                let b = root(&mut parents, prior.label);
                parents[a.max(b)] = a.min(b);
            }
            current.push(Run {
                start,
                end: x,
                label,
            });
        }
        std::mem::swap(&mut previous, &mut current);
        current.clear();
    }
    // Each run initially contributes one row. Collapse bounds and pixel counts
    // into its root; roots always precede their children.
    for i in (0..runs.len()).rev() {
        let r = root(&mut parents, i);
        if r == i {
            continue;
        }
        let (x, y, w, h) = runs[i].bbox;
        let pixels = runs[i].pixels;
        let target = &mut runs[r];
        let right = (target.bbox.0 + target.bbox.2).max(x + w);
        let bottom = (target.bbox.1 + target.bbox.3).max(y + h);
        target.bbox.0 = target.bbox.0.min(x);
        target.bbox.1 = target.bbox.1.min(y);
        target.bbox.2 = right - target.bbox.0;
        target.bbox.3 = bottom - target.bbox.1;
        target.pixels += pixels;
        runs[i].pixels = 0;
    }
    runs.retain(|c| c.pixels != 0);
    runs
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Luma;
    use imageproc::{
        distance_transform::Norm, morphology::close, region_labelling::connected_components,
    };
    use std::collections::BTreeMap;

    #[test]
    fn binary_operations_match_reference_on_borders_and_noise() {
        let mut state = 12u32;
        for (w, h) in [(0, 0), (1, 1), (2, 13), (65, 33), (128, 96)] {
            for density in [0, 20, 128, 235, 255] {
                let image = GrayImage::from_fn(w, h, |_, _| {
                    state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                    Luma([if (state >> 24) < density { 255 } else { 0 }])
                });
                for radius in [1, 2] {
                    assert_eq!(
                        close_binary(&image, radius),
                        close(&image, Norm::LInf, radius as u8)
                    );
                }
                for connectivity in [Connectivity::Four, Connectivity::Eight] {
                    for inverse in [false, true] {
                        // The reference labeler cannot represent a foreground
                        // label in its one-pixel union-find allocation.
                        if w * h == 1 {
                            let actual = component_boxes(&image, connectivity, inverse);
                            assert_eq!(
                                actual.len(),
                                usize::from((image.as_raw()[0] == 0) == inverse)
                            );
                            continue;
                        }
                        let labels = connected_components(
                            &image,
                            connectivity,
                            Luma([if inverse { 255 } else { 0 }]),
                        );
                        let mut reference: BTreeMap<u32, Vec<(u32, u32)>> = BTreeMap::new();
                        for (x, y, label) in labels.enumerate_pixels() {
                            if label[0] != 0 {
                                reference.entry(label[0]).or_default().push((x, y));
                            }
                        }
                        let mut expected: Vec<_> = reference
                            .values()
                            .map(|points| {
                                let x = points.iter().map(|p| p.0).min().unwrap();
                                let y = points.iter().map(|p| p.1).min().unwrap();
                                (
                                    (
                                        x,
                                        y,
                                        points.iter().map(|p| p.0).max().unwrap() - x + 1,
                                        points.iter().map(|p| p.1).max().unwrap() - y + 1,
                                    ),
                                    points.len() as u32,
                                )
                            })
                            .collect();
                        let mut actual: Vec<_> = component_boxes(&image, connectivity, inverse)
                            .iter()
                            .map(|c| (c.bbox, c.pixels))
                            .collect();
                        actual.sort_unstable();
                        expected.sort_unstable();
                        assert_eq!(actual, expected);
                    }
                }
            }
        }
    }
}
