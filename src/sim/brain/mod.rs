mod gru;

pub use self::gru::{InputVector, OutputVector};

use std::sync::Arc;

use na;

pub type OutLen = na::dimension::U24;
// 1 + 8 + 8 + 8
pub type InLen = na::dimension::U25;

#[derive(Clone)]
pub struct Brain {
    gru: Arc<gru::MGRU>,
    pub signal: f32,
    hiddens: OutputVector,
}

impl Default for Brain {
    fn default() -> Brain {
        Brain {
            gru: Arc::new(gru::MGRU::new_rand()),
            signal: 0.0,
            hiddens: OutputVector::new_random(),
        }
    }
}

impl Brain {
    pub fn mutate(&mut self, lambda: f64) {
        if let Some(gru) = self.gru.mutated(lambda) {
            self.gru = Arc::new(gru);
        }
    }

    pub fn apply(&mut self, inputs: &gru::InputVector) -> gru::OutputVector {
        self.hiddens = self.gru.apply(inputs, &self.hiddens);
        self.hiddens
    }
}
