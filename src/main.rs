#![feature(plugin)]
#![plugin(clippy)]

extern crate arrayfire as af;
extern crate gridsim as gs;
extern crate gridsim_ui as ui;
extern crate noisy_float;
extern crate rand;

mod sim;

use sim::E12;

const MAX_FOOD_DISPLAYED: usize = 20;

fn main() {
    ui::run::basic(gs::SquareGrid::<E12>::new(128, 128), |c| {
        use std::cmp::min;
        let color = min(c.food, MAX_FOOD_DISPLAYED) as f32 / MAX_FOOD_DISPLAYED as f32;
        [color, color, color, 1.0]
    });
}
