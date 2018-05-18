extern crate boolinator;
extern crate gridsim as gs;
extern crate gridsim_ui as ui;
extern crate nalgebra as na;
extern crate noisy_float;
extern crate rand;

mod sim;

use sim::E12;

use std::f32::consts::PI;

const MAX_FOOD_DISPLAYED: usize = 256;
const COLOR_CYCLES: f32 = 3.0;
const COLOR_SHIFT: f32 = PI + 0.8;

//const DIMS: (usize, usize) = (256, 144);
const DIMS: (usize, usize) = (426, 240);
//const DIMS: (usize, usize) = (640, 360);

fn rgb(n: f32) -> (f32, f32, f32) {
    let n = n * 0.5 + 0.5;
    let angle = n * COLOR_CYCLES * PI + COLOR_SHIFT;
    let ratio = 3.0 * PI / 4.0;
    let colorsin = |a: f32| a.sin() * 0.5 + 0.5;
    (
        colorsin(angle),
        colorsin(angle + ratio),
        colorsin(angle + 2.0 * ratio),
    )
}

fn main() {
    ui::run::basic(gs::SquareGrid::<E12>::new(DIMS.0, DIMS.1), |c| {
        use std::cmp::min;
        let intensity =
            0.015f32.max(min(c.food, MAX_FOOD_DISPLAYED) as f32 / MAX_FOOD_DISPLAYED as f32);
        let hue = c.brain
            .as_ref()
            .map(|b| rgb(b.signal))
            .unwrap_or((0.0, 0.0, 0.0));
        [hue.0 * intensity, hue.1 * intensity, hue.2 * intensity, 1.0]
    });
}
