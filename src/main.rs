#![feature(plugin)]
#![plugin(clippy)]

extern crate arrayfire as af;
extern crate gridsim as gs;
extern crate gridsim_ui as ui;
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
    af::set_backend(af::Backend::CPU);
    ui::run::basic(gs::SquareGrid::<E12>::new(64, 64), |c| {
        use std::cmp::min;
        let intensity = min(c.food, MAX_FOOD_DISPLAYED) as f32 / MAX_FOOD_DISPLAYED as f32;
        let hue = rgb(c.brain.as_ref().map(|b| b.signal).unwrap_or(0.0));
        [hue.0 * intensity, hue.1 * intensity, hue.2 * intensity, 1.0]
    });
}
