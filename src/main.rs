extern crate gridsim as gs;
extern crate gridsim_ui as ui;
extern crate nalgebra as na;
extern crate noisy_float;
extern crate rand;

mod sim;

use sim::E12;

use std::f32::consts::PI;

const MAX_FOOD_DISPLAYED: usize = 4 * sim::SPAWN_FOOD;
const COLOR_CYCLES: f32 = 2.0;

pub fn sigmoid(n: f32) -> f32 {
    (1.0 + (-n).exp()).recip()
}

fn rgb(n: f32) -> (f32, f32, f32) {
    let n = sigmoid(n);
    let angle = n * COLOR_CYCLES * PI;
    let ratio = 3.0 * PI / 4.0;
    let colorsin = |a: f32| a.sin() * 0.5 + 0.5;
    (
        colorsin(angle),
        colorsin(angle + ratio),
        colorsin(angle + 2.0 * ratio),
    )
}

fn main() {
    ui::run::basic(gs::SquareGrid::<E12>::new(426, 240), |c| {
        use std::cmp::min;
        let intensity =
            0.005f32.max(min(c.food, MAX_FOOD_DISPLAYED) as f32 / MAX_FOOD_DISPLAYED as f32);
        let hue = c.brain
            .as_ref()
            .map(|b| rgb(b.signal))
            .unwrap_or((0.0, 0.0, 0.0));
        [hue.0 * intensity, hue.1 * intensity, hue.2 * intensity, 1.0]
    });
}
