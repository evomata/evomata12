extern crate gridsim as gs;
extern crate gridsim_ui as ui;

mod sim;
use sim::E12;

fn main() {
    ui::run::basic_par(gs::Grid::<E12>::new(512, 512), |c| [1.0, 1.0, 1.0, 1.0]);
}
