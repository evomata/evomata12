extern crate array_init;
extern crate boolinator;
extern crate gridsim as gs;
extern crate gridsim_ui as ui;
extern crate nalgebra as na;
extern crate noisy_float;
extern crate os_pipe;
extern crate rand;
extern crate serde;
#[macro_use]
extern crate serde_derive;

use os_pipe::pipe;

mod sim;

use sim::E12;

use std::f32::consts::PI;

const MAX_FOOD_DISPLAYED: usize = 512;
const COLOR_CYCLES: f32 = 3.0;
const COLOR_SHIFT: f32 = PI + 0.8;

const DIMS: (usize, usize) = (256, 144);
//const DIMS: (usize, usize) = (426, 240);
//const DIMS: (usize, usize) = (640, 360);
//const DIMS: (usize, usize) = (768, 432);
//const DIMS: (usize, usize) = (800, 450);
//const DIMS: (usize, usize) = (896, 504);
//const DIMS: (usize, usize) = (960, 540);
//const DIMS: (usize, usize) = (1280, 720);

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
    let (in_right, out_left) = pipe().unwrap();
    let (in_up_right, out_down_left) = pipe().unwrap();
    let (in_up, out_down) = pipe().unwrap();
    let (in_up_left, out_down_right) = pipe().unwrap();
    let (in_left, out_right) = pipe().unwrap();
    let (in_down_left, out_up_right) = pipe().unwrap();
    let (in_down, out_up) = pipe().unwrap();
    let (in_down_right, out_up_left) = pipe().unwrap();
    unsafe {
        ui::Loop::new(|c: &sim::Cell| {
            use std::cmp::min;
            let intensity =
                0.015f32.max(min(c.food, MAX_FOOD_DISPLAYED) as f32 / MAX_FOOD_DISPLAYED as f32);
            let hue = c
                .brain
                .as_ref()
                .map(|b| rgb(b.signal))
                .unwrap_or((0.0, 0.0, 0.0));
            [hue.0 * intensity, hue.1 * intensity, hue.2 * intensity, 1.0]
        }).run_multi(
            gs::SquareGrid::<E12>::new(DIMS.0, DIMS.1),
            in_right,
            in_up_right,
            in_up,
            in_up_left,
            in_left,
            in_down_left,
            in_down,
            in_down_right,
            out_right,
            out_up_right,
            out_up,
            out_up_left,
            out_left,
            out_down_left,
            out_down,
            out_down_right,
        );
    }
}
