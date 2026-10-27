//! Pure computer-vision detector (always-available fallback; `image` +
//! `imageproc` only). Pipeline: downscale → grayscale → Canny → morphological
//! close → connected components → shape filtering/scoring → drop icon-in-hit-area
//! nesting. A sensitive edge pass recovers faint outlined fields. Sees visual
//! structure, not semantic clickability.

use anyhow::{bail, Context};
use image::{GrayImage, Luma};
use imageproc::{
    distance_transform::Norm,
    edges::canny,
    morphology::close,
    region_labelling::{connected_components, Connectivity},
};
use std::collections::HashMap;

use super::detect::{DetectorOutput, HintCandidate, HintDetector};
use crate::{
    backend::Backend,
    config::{config, HintConfig},
};

#[derive(Default)]
pub struct CvHintDetector;

impl HintDetector for CvHintDetector {
    fn name(&self) -> &'static str {
        "cv"
    }

    fn detect(&self, backend: &mut dyn Backend) -> anyhow::Result<DetectorOutput> {
        let capture = backend.capture_screen().context(
            "hint mode requires screen capture; the CV detector cannot run on this backend",
        )?;
        let candidates = detect_candidates_cv(&capture.bgra, capture.w, capture.h)?;
        Ok(DetectorOutput {
            candidates,
            capture_w: capture.w,
            capture_h: capture.h,
            focus_rect: None,
        })
    }
}

/// A component in the downscaled working image: its box plus how many mask
/// pixels it actually occupies (used for the solidity score).
struct Component {
    bbox: (u32, u32, u32, u32),
    pixels: u32,
}

fn detect_candidates_cv(
    bgra: &[u8],
    capture_w: u32,
    capture_h: u32,
) -> anyhow::Result<Vec<HintCandidate>> {
    detect_candidates_with_config(bgra, capture_w, capture_h, &config().hint)
}

fn detect_candidates_with_config(
    bgra: &[u8],
    capture_w: u32,
    capture_h: u32,
    cfg: &HintConfig,
) -> anyhow::Result<Vec<HintCandidate>> {
    if capture_w == 0 || capture_h == 0 {
        bail!("cannot run hint CV detector on an empty capture");
    }
    if bgra.len() < (capture_w as usize * capture_h as usize * 4) {
        bail!("screen capture buffer is smaller than declared dimensions");
    }

    let longest = capture_w.max(capture_h);
    let scale = if cfg.downscale_longest > 0 && longest > cfg.downscale_longest {
        cfg.downscale_longest as f32 / longest as f32
    } else {
        1.0
    };
    let ww = ((capture_w as f32 * scale).round() as u32).max(1);
    let wh = ((capture_h as f32 * scale).round() as u32).max(1);

    let gray = grayscale_resized(bgra, capture_w as usize, capture_h as usize, ww, wh);
    let edges = canny(
        &gray,
        cfg.edge_low_threshold as f32,
        cfg.edge_high_threshold as f32,
    );
    // Closing (dilate + erode) reconnects the broken strokes of one element
    // without inflating boxes or bridging neighbours.
    const CLOSE_RADIUS: u8 = 2;
    let closed = close(&edges, Norm::LInf, CLOSE_RADIUS);

    // Faint input borders often have no edge strong enough to seed the normal
    // Canny pass. Restrict this sensitive pass to field outlines and small,
    // compact icons so it does not label every faint glyph or separator.
    let faint_edges = canny(
        &gray,
        cfg.edge_low_threshold as f32 / 32.0,
        cfg.edge_high_threshold as f32 / 32.0,
    );
    // A larger radius bridges placeholder glyphs to short field borders and
    // merges fields near toolbar edges (especially after downscaling).
    let faint_closed = close(&faint_edges, Norm::LInf, 1);
    let mut components = connected_component_boxes(&closed);
    components.extend(
        connected_component_boxes(&faint_closed)
            .into_iter()
            .filter(|c| {
                field_outline_support(&faint_closed, c.bbox).is_some() || is_compact_icon(c)
            })
            .filter(|c| !components.iter().any(|existing| existing.bbox == c.bbox))
            .collect::<Vec<_>>(),
    );

    // Closing can join a field's border to adjacent toolbar/panel edges. Its
    // foreground component then describes the whole panel, not the field.
    // The enclosed background still describes the field's interior.
    for field in enclosed_field_boxes(&faint_closed) {
        if !components.iter().any(|existing| {
            let a = existing.bbox;
            let b = field.bbox;
            a.0.abs_diff(b.0) <= 3
                && a.1.abs_diff(b.1) <= 3
                && a.2.abs_diff(b.2) <= 6
                && a.3.abs_diff(b.3) <= 6
        }) {
            components.push(field);
        }
    }

    // Shape filters, relative to the working image so behaviour matches at
    // 1080p and 4K: minimum side (sub-glyph noise), maximum side/area
    // fractions (bars, panels, banners), maximum elongation (separators,
    // scrollbars, underlines).
    const MIN_SIZE: u32 = 8;
    const MAX_SIZE_FRAC: f32 = 0.6;
    const MAX_AREA_FRAC: f32 = 0.25;
    const MAX_ASPECT: f32 = 12.0;

    let work_area = (ww as f32) * (wh as f32);
    let max_w = (MAX_SIZE_FRAC * ww as f32).max(MIN_SIZE as f32);
    let max_h = (MAX_SIZE_FRAC * wh as f32).max(MIN_SIZE as f32);
    let max_area = (MAX_AREA_FRAC * work_area).max(1.0);

    let kept: Vec<(Component, f32)> = components
        .into_iter()
        .filter_map(|component| {
            let (_, _, w, h) = component.bbox;
            if w < MIN_SIZE || h < MIN_SIZE {
                return None;
            }
            let field_support = field_outline_support(&faint_closed, component.bbox);
            let width_limit = if field_support.is_some() {
                0.9 * ww as f32
            } else {
                max_w
            };
            if (w as f32) > width_limit || (h as f32) > max_h {
                return None;
            }
            let area = (w as f32) * (h as f32);
            if area > max_area {
                return None;
            }
            let aspect = (w.max(h) as f32) / (w.min(h).max(1) as f32);
            if aspect > MAX_ASPECT && field_support.is_none() {
                return None;
            }
            // Coherent controls fill their box; gradient/text speckle scores low.
            let solidity = (component.pixels as f32 / area).clamp(0.0, 1.0);
            // Hollow field borders have low fill despite being coherent targets.
            Some((component, field_support.unwrap_or(solidity)))
        })
        .collect();

    let kept = drop_icon_inner_boxes(kept);

    let inv_scale = 1.0 / scale;
    Ok(kept
        .into_iter()
        .map(|(component, solidity)| {
            let bbox = scale_box_to_capture(component.bbox, inv_scale, capture_w, capture_h);
            HintCandidate {
                bbox,
                score: solidity,
            }
        })
        .collect())
}

/// Recover small muted icons from the sensitive pass. At working resolution,
/// individual text strokes are usually narrower than eight pixels; elongated
/// text groups and sparse edge fragments must not qualify as compact icons.
fn is_compact_icon(component: &Component) -> bool {
    let (_, _, w, h) = component.bbox;
    let short = w.min(h);
    let long = w.max(h);
    (8..=32).contains(&short)
        && long <= 32
        && long * 2 <= short * 3
        && component.pixels as f32 / (w * h) as f32 >= 0.55
}

/// Find enclosed interiors independently of connected foreground borders.
/// Keep only mostly empty, field-shaped regions surrounded on all four sides;
/// screen edges and open outlines must not manufacture an enclosure.
fn enclosed_field_boxes(mask: &GrayImage) -> Vec<Component> {
    let inverse = GrayImage::from_fn(mask.width(), mask.height(), |x, y| {
        Luma([if mask.get_pixel(x, y)[0] == 0 { 255 } else { 0 }])
    });
    // Foreground edges use eight neighbours; background must use four so a
    // diagonal rounded corner stays closed rather than leaking outside.
    component_boxes_with_connectivity(&inverse, Connectivity::Four)
        .into_iter()
        .filter_map(|mut component| {
            let (x, y, w, h) = component.bbox;
            if x == 0 || y == 0 || x + w >= mask.width() || y + h >= mask.height() {
                return None;
            }
            if component.pixels as f32 / ((w * h) as f32) < 0.6 {
                return None;
            }
            // Include the immediately surrounding edge, so the normal outline
            // validator checks the enclosure rather than its empty interior.
            component.bbox = (x - 1, y - 1, w + 2, h + 2);
            field_outline_support(mask, component.bbox)?;
            Some(component)
        })
        .collect()
}

/// Evidence for an outlined, horizontal field. Require most of every side;
/// long rules, text lines, and open shapes must not become wide input targets.
/// Ignore a few corner pixels to allow rounded borders, and tolerate a small
/// inset because Canny can produce edges on either side of a border.
fn field_outline_support(mask: &GrayImage, (x, y, w, h): (u32, u32, u32, u32)) -> Option<f32> {
    if !(8..=96).contains(&h) || w < h * 2 || w > h * 40 {
        return None;
    }
    let margin = 4.min(h / 4);
    let horizontal = |side_y: u32| {
        (x + margin..x + w - margin)
            .filter(|&px| {
                (side_y.saturating_sub(2).max(y)..=(side_y + 2).min(y + h - 1))
                    .any(|py| mask.get_pixel(px, py)[0] != 0)
            })
            .count() as f32
            / (w - 2 * margin) as f32
    };
    let vertical = |side_x: u32| {
        (y + margin..y + h - margin)
            .filter(|&py| {
                (side_x.saturating_sub(2).max(x)..=(side_x + 2).min(x + w - 1))
                    .any(|px| mask.get_pixel(px, py)[0] != 0)
            })
            .count() as f32
            / (h - 2 * margin) as f32
    };
    let support = horizontal(y)
        .min(horizontal(y + h - 1))
        .min(vertical(x))
        .min(vertical(x + w - 1));
    (support >= 0.7).then_some(support)
}

/// Map a working-image box back to clamped capture-pixel coordinates.
fn scale_box_to_capture(
    (x, y, w, h): (u32, u32, u32, u32),
    inv_scale: f32,
    capture_w: u32,
    capture_h: u32,
) -> (u32, u32, u32, u32) {
    let nx = ((x as f32 * inv_scale).round() as u32).min(capture_w.saturating_sub(1));
    let ny = ((y as f32 * inv_scale).round() as u32).min(capture_h.saturating_sub(1));
    let nw = ((w as f32 * inv_scale).round() as u32)
        .max(1)
        .min(capture_w - nx);
    let nh = ((h as f32 * inv_scale).round() as u32)
        .max(1)
        .min(capture_h - ny);
    (nx, ny, nw, nh)
}

/// Drop a box sitting concentrically inside a small, near-square box — an
/// icon glyph inside its own hit area, where both labels would point at the
/// same place. Deliberately narrow so container children are preserved.
/// Thresholds are in working-image pixels.
fn drop_icon_inner_boxes(mut items: Vec<(Component, f32)>) -> Vec<(Component, f32)> {
    const MAX_PARENT_SIDE: u32 = 48;
    const SQUARE_TOLERANCE: u32 = 6;
    const CONCENTRIC_TOLERANCE: u32 = 6;

    let boxes: Vec<(u32, u32, u32, u32)> = items.iter().map(|(c, _)| c.bbox).collect();
    let mut drop = vec![false; boxes.len()];

    for (i, &child) in boxes.iter().enumerate() {
        let smallest_parent = boxes
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != i && !drop[*j])
            .filter(|(_, parent)| contains(**parent, child))
            .min_by_key(|(_, parent)| parent.2 * parent.3)
            .map(|(_, parent)| *parent);

        let Some(parent) = smallest_parent else {
            continue;
        };
        let near_square = parent.2.abs_diff(parent.3) <= SQUARE_TOLERANCE;
        let small = parent.2 <= MAX_PARENT_SIDE && parent.3 <= MAX_PARENT_SIDE;
        let child_center = (child.0 + child.2 / 2, child.1 + child.3 / 2);
        let parent_center = (parent.0 + parent.2 / 2, parent.1 + parent.3 / 2);
        let concentric = child_center.0.abs_diff(parent_center.0) <= CONCENTRIC_TOLERANCE
            && child_center.1.abs_diff(parent_center.1) <= CONCENTRIC_TOLERANCE;
        if small && near_square && concentric {
            drop[i] = true;
        }
    }

    let mut keep = drop.into_iter();
    items.retain(|_| !keep.next().unwrap_or(false));
    items
}

fn contains(parent: (u32, u32, u32, u32), child: (u32, u32, u32, u32)) -> bool {
    parent.0 <= child.0
        && parent.1 <= child.1
        && parent.0 + parent.2 >= child.0 + child.2
        && parent.1 + parent.3 >= child.1 + child.3
}

/// Nearest-neighbour downscale of a BGRA capture straight into grayscale —
/// edges survive it, and it is far cheaper than a filtered resize.
fn grayscale_resized(bgra: &[u8], src_w: usize, src_h: usize, dst_w: u32, dst_h: u32) -> GrayImage {
    let dst_w_usize = dst_w as usize;
    let dst_h_usize = dst_h as usize;
    let mut out = vec![0u8; dst_w_usize * dst_h_usize];
    for y in 0..dst_h_usize {
        let src_y = (y * src_h / dst_h_usize).min(src_h - 1);
        for x in 0..dst_w_usize {
            let src_x = (x * src_w / dst_w_usize).min(src_w - 1);
            let off = (src_y * src_w + src_x) * 4;
            let b = bgra[off] as u32;
            let g = bgra[off + 1] as u32;
            let r = bgra[off + 2] as u32;
            out[y * dst_w_usize + x] = ((29 * b + 150 * g + 77 * r) / 256) as u8;
        }
    }
    GrayImage::from_vec(dst_w, dst_h, out).expect("grayscale buffer size matches image dimensions")
}

/// Connected components of a binary mask, with occupied-pixel counts for the
/// solidity score.
fn connected_component_boxes(mask: &GrayImage) -> Vec<Component> {
    component_boxes_with_connectivity(mask, Connectivity::Eight)
}

fn component_boxes_with_connectivity(
    mask: &GrayImage,
    connectivity: Connectivity,
) -> Vec<Component> {
    let labels = connected_components(mask, connectivity, Luma([0u8]));
    // (min_x, min_y, max_x, max_y, pixel_count)
    let mut rects: HashMap<u32, (u32, u32, u32, u32, u32)> = HashMap::new();

    for (x, y, label) in labels.enumerate_pixels() {
        let label = label[0];
        if label == 0 {
            continue;
        }
        rects
            .entry(label)
            .and_modify(|rect| {
                rect.0 = rect.0.min(x);
                rect.1 = rect.1.min(y);
                rect.2 = rect.2.max(x);
                rect.3 = rect.3.max(y);
                rect.4 += 1;
            })
            .or_insert((x, y, x, y, 1));
    }

    rects
        .into_values()
        .map(|(min_x, min_y, max_x, max_y, pixels)| Component {
            bbox: (min_x, min_y, max_x - min_x + 1, max_y - min_y + 1),
            pixels,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn detect_gray(img: &GrayImage, cfg: &HintConfig) -> Vec<HintCandidate> {
        let bgra: Vec<u8> = img.pixels().flat_map(|p| [p[0], p[0], p[0], 255]).collect();
        detect_candidates_with_config(&bgra, img.width(), img.height(), cfg).unwrap()
    }

    #[test]
    fn detects_faint_and_wide_input_outlines() {
        // Light/dark themes, rounded corners, and a high-DPI capture. At 20:1
        // these fields were also rejected by the generic 12:1 aspect filter.
        for (background, border, factor, width) in [
            (255, 235, 1, 480),
            (30, 50, 1, 480),
            (255, 235, 2, 480),
            (255, 235, 1, 200),
            (255, 0, 1, 480),
        ] {
            let mut img = GrayImage::from_pixel(640 * factor, 360 * factor, Luma([background]));
            let expected = (40 * factor, 100 * factor, width * factor, 24 * factor);
            let (x, y, w, h) = expected;
            for py in 0..h {
                for px in 0..w {
                    let dx = px.min(w - 1 - px) / factor;
                    let dy = py.min(h - 1 - py) / factor;
                    // A one-logical-pixel outline with clipped rounded corners.
                    if (dy == 0 && dx >= 4)
                        || (dx == 0 && dy >= 4)
                        || (dx + dy == 3 && dx > 0 && dy > 0)
                    {
                        img.put_pixel(x + px, y + py, Luma([border]));
                    }
                }
            }
            // Placeholder-like strokes must not obscure the field's border.
            for px in (x + 12 * factor..x + 100 * factor).step_by(8 * factor as usize) {
                for py in y + 8 * factor..y + 16 * factor {
                    img.put_pixel(px, py, Luma([border]));
                }
            }
            let cfg = HintConfig {
                downscale_longest: 640,
                ..HintConfig::default()
            };
            let candidates = detect_gray(&img, &cfg);
            assert!(
                candidates.iter().any(|c| {
                    c.bbox.0.abs_diff(x) <= 4 * factor
                        && c.bbox.1.abs_diff(y) <= 4 * factor
                        && c.bbox.2.abs_diff(w) <= 4 * factor
                        && c.bbox.3.abs_diff(h) <= 4 * factor
                }),
                "missing {expected:?} on background {background}: {candidates:?}"
            );
        }
    }

    #[test]
    fn detects_inputs_connected_to_panel_edges() {
        for (background, field) in [(26, 34), (245, 237), (26, 31)] {
            let mut img = GrayImage::from_pixel(640, 360, Luma([background]));
            // Filled composer with a bottom border shared with a panel rule.
            // The foreground component spans the screen and is not field-shaped.
            for y in 280..308 {
                for x in 100..580 {
                    img.put_pixel(x, y, Luma([field]));
                }
            }
            for x in 0..640 {
                img.put_pixel(x, 308, Luma([field]));
            }
            // Text and action icons leave the interior mostly connected.
            for x in (112..200).step_by(8) {
                for y in 290..298 {
                    img.put_pixel(x, y, Luma([128]));
                }
            }
            let candidates = detect_gray(&img, &HintConfig::default());
            assert!(
                candidates.iter().any(|c| {
                    c.bbox.0.abs_diff(100) <= 3
                        && c.bbox.1.abs_diff(280) <= 3
                        && c.bbox.2.abs_diff(480) <= 6
                        && c.bbox.3.abs_diff(28) <= 6
                }),
                "missing connected composer on {background}: {candidates:?}"
            );
        }
    }

    #[test]
    fn enclosed_fields_preserve_diagonal_corners_and_reject_open_regions() {
        let mut mask = GrayImage::new(640, 360);
        // Diamond corners: eight-connected foreground encloses four-connected
        // background, as on rounded fields after Canny and downscaling.
        for x in 44..516 {
            mask.put_pixel(x, 100, Luma([255]));
            mask.put_pixel(x, 124, Luma([255]));
        }
        for y in 104..121 {
            mask.put_pixel(40, y, Luma([255]));
            mask.put_pixel(519, y, Luma([255]));
        }
        for d in 0..4 {
            for (x, y) in [
                (40 + d, 104 - d),
                (519 - d, 104 - d),
                (40 + d, 120 + d),
                (519 - d, 120 + d),
            ] {
                mask.put_pixel(x, y, Luma([255]));
            }
        }
        // Join the outline to a panel border; the interior must still survive.
        for x in 0..44 {
            mask.put_pixel(x, 100, Luma([255]));
        }
        let fields = enclosed_field_boxes(&mask);
        assert_eq!(fields.len(), 1);
        assert_eq!(fields[0].bbox, (40, 100, 480, 25));
        // A real gap must leak outside, unlike a diagonal corner.
        mask.put_pixel(40, 112, Luma([0]));
        assert!(enclosed_field_boxes(&mask).is_empty());
    }

    #[test]
    fn detects_muted_tab_icons_after_downscaling() {
        for (background, stroke, factor) in [(38, 80, 1), (38, 80, 2), (240, 198, 1)] {
            let mut img = GrayImage::from_pixel(1280 * factor, 720 * factor, Luma([background]));
            // A small ring with an internal handset-like stroke. Its contrast
            // cannot seed the normal Canny pass, as with muted tab favicons.
            for y in 0..11 * factor {
                for x in 0..11 * factor {
                    let dx = x as f32 / factor as f32 - 5.0;
                    let dy = y as f32 / factor as f32 - 5.0;
                    let r = dx * dx + dy * dy;
                    let ring = (12.0..=25.0).contains(&r);
                    let handset = (-2.0..=-1.0).contains(&dx) && (-2.0..=1.0).contains(&dy)
                        || (-1.0..=2.0).contains(&dx) && (1.0..=2.0).contains(&dy);
                    if ring || handset {
                        img.put_pixel(200 * factor + x, 20 * factor + y, Luma([stroke]));
                    }
                }
            }
            let candidates = detect_gray(&img, &HintConfig::default());
            assert!(
                candidates.iter().any(|c| {
                    c.bbox.0.abs_diff(200 * factor) <= 2 * factor
                        && c.bbox.1.abs_diff(20 * factor) <= 2 * factor
                        && c.bbox.2.abs_diff(11 * factor) <= 4 * factor
                        && c.bbox.3.abs_diff(11 * factor) <= 4 * factor
                }),
                "missing muted icon at scale {factor} on {background}: {candidates:?}"
            );
        }
    }

    #[test]
    fn sensitive_pass_rejects_rules_and_open_outlines() {
        let mut img = GrayImage::from_pixel(640, 360, Luma([255]));
        for x in 40..520 {
            img.put_pixel(x, 100, Luma([235]));
            img.put_pixel(x, 180, Luma([235]));
            img.put_pixel(x, 204, Luma([235]));
        }
        for y in 180..205 {
            img.put_pixel(40, y, Luma([235]));
        }
        assert!(detect_gray(&img, &HintConfig::default()).is_empty());
    }

    #[test]
    fn solidity_is_pixel_fill_ratio() {
        // A hollow 4x4 ring: 12 of 16 box pixels filled, so solidity = 12/16.
        let mut img = GrayImage::new(4, 4);
        for (x, y, p) in img.enumerate_pixels_mut() {
            if x == 0 || x == 3 || y == 0 || y == 3 {
                *p = Luma([255]);
            }
        }
        let comps = connected_component_boxes(&img);
        assert_eq!(comps.len(), 1);
        assert_eq!(comps[0].pixels, 12);
        assert_eq!(comps[0].bbox, (0, 0, 4, 4));
        let (_, _, w, h) = comps[0].bbox;
        let solidity = comps[0].pixels as f32 / (w * h) as f32;
        assert!((solidity - 12.0 / 16.0).abs() < 1e-6);
    }

    #[test]
    fn icon_inner_box_is_dropped_but_container_children_survive() {
        let comp = |bbox: (u32, u32, u32, u32)| {
            (
                Component {
                    bbox,
                    pixels: bbox.2 * bbox.3,
                },
                1.0,
            )
        };
        // Small near-square icon (outer ring) with a concentric inner glyph.
        let icon_outer = comp((100, 100, 24, 24));
        let icon_inner = comp((106, 106, 12, 12));
        // A large container with two real buttons inside it.
        let container = comp((0, 0, 400, 120));
        let button_a = comp((10, 20, 80, 30));
        let button_b = comp((120, 20, 80, 30));

        let kept =
            drop_icon_inner_boxes(vec![icon_outer, icon_inner, container, button_a, button_b]);
        let boxes: Vec<_> = kept.iter().map(|(c, _)| c.bbox).collect();

        // The concentric inner glyph is gone.
        assert!(!boxes.contains(&(106, 106, 12, 12)));
        // Everything else — including the container's children — is preserved.
        assert!(boxes.contains(&(100, 100, 24, 24)));
        assert!(boxes.contains(&(0, 0, 400, 120)));
        assert!(boxes.contains(&(10, 20, 80, 30)));
        assert!(boxes.contains(&(120, 20, 80, 30)));
    }
}

/// Dev tool, not a test of invariants: run the CV detector over a real
/// screenshot and print every candidate box, to study behaviour on actual UIs:
///   grim -t ppm /tmp/shot.ppm
///   STOCHOS_CV_IMAGE=/tmp/shot.ppm cargo test --release -- --ignored --nocapture cv_boxes
#[cfg(test)]
mod probe {
    use super::*;

    /// Minimal binary PPM (P6, maxval 255) reader → BGRA buffer.
    fn read_ppm(path: &str) -> (Vec<u8>, u32, u32) {
        let data = std::fs::read(path).expect("read image file");
        let mut fields = Vec::new();
        let mut pos = 0;
        while fields.len() < 4 {
            // Skip whitespace and comments between header fields.
            while pos < data.len() && (data[pos].is_ascii_whitespace() || data[pos] == b'#') {
                if data[pos] == b'#' {
                    while pos < data.len() && data[pos] != b'\n' {
                        pos += 1;
                    }
                }
                pos += 1;
            }
            let start = pos;
            while pos < data.len() && !data[pos].is_ascii_whitespace() {
                pos += 1;
            }
            fields.push(std::str::from_utf8(&data[start..pos]).unwrap().to_owned());
        }
        pos += 1; // single whitespace after maxval
        assert_eq!(fields[0], "P6", "expected binary PPM (grim -t ppm)");
        let (w, h): (u32, u32) = (fields[1].parse().unwrap(), fields[2].parse().unwrap());
        let rgb = &data[pos..pos + (w * h * 3) as usize];
        let mut bgra = vec![255u8; (w * h * 4) as usize];
        for i in 0..(w * h) as usize {
            bgra[i * 4] = rgb[i * 3 + 2];
            bgra[i * 4 + 1] = rgb[i * 3 + 1];
            bgra[i * 4 + 2] = rgb[i * 3];
        }
        (bgra, w, h)
    }

    #[test]
    #[ignore = "set STOCHOS_CV_IMAGE to a P6 .ppm screenshot"]
    fn cv_boxes_from_screenshot() {
        let path = std::env::var("STOCHOS_CV_IMAGE").expect("set STOCHOS_CV_IMAGE");
        std::env::set_var(
            "XDG_CONFIG_HOME",
            std::env::temp_dir().join("stochos-cv-probe"),
        );
        crate::config::init();
        let (bgra, w, h) = read_ppm(&path);
        let start = std::time::Instant::now();
        let candidates = detect_candidates_cv(&bgra, w, h).expect("cv detection");
        eprintln!(
            "cv: {} candidates in {:?} on {w}x{h}",
            candidates.len(),
            start.elapsed()
        );
        let mut sorted = candidates;
        sorted.sort_by_key(|c| (c.bbox.1, c.bbox.0));
        for c in &sorted {
            eprintln!("  bbox={:?} score={:.2}", c.bbox, c.score);
        }
    }
}
