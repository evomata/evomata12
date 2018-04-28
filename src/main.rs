extern crate arrayfire as af;
extern crate gridsim as gs;
extern crate gridsim_ui as ui;
#[macro_use]
extern crate enum_iterator_derive;

mod sim;

use sim::E12;

fn main() {
    ui::run::basic_par(gs::Grid::<E12>::new(512, 512), |_| [1.0, 1.0, 1.0, 1.0]);
}
