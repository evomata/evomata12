mod gru;

pub use self::gru::{InputVector, OutputVector};

use na;

pub type OutLen = na::dimension::U24;
// 1 + 8 + 8 + 8
pub type InLen = na::dimension::U25;

#[derive(Clone)]
pub struct Brain {
    gru: Box<gru::MGRU>,
    pub signal: f32,
}

impl Default for Brain {
    fn default() -> Brain {
        Brain {
            gru: Box::new(gru::MGRU::new_rand()),
            signal: 0.0,
        }
    }
}

impl Brain {
    pub fn mutate(&mut self, lambda: f64) {
        self.gru.mutate(lambda);
    }

    pub fn apply(&mut self, inputs: &gru::InputVector) -> gru::OutputVector {
        self.gru.apply(inputs)
    }
}
