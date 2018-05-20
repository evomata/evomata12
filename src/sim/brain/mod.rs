mod gru;

pub use self::gru::{InputVector, OutputVector};

use array_init;
use na;
use std::sync::Arc;

pub type OutLen = na::dimension::U16;
// 1 + 8 + 8 + 8
pub type InLen = na::dimension::U25;

// Number of extra layers
const EXTRA_LAYERS: usize = 1;

#[derive(Clone)]
pub struct Brain {
    input_gru: Arc<gru::MGRUInput>,
    input_hiddens: OutputVector,
    internal_grus: [Arc<gru::MGRU>; EXTRA_LAYERS],
    internal_hiddens: [OutputVector; EXTRA_LAYERS],
    pub signal: f32,
}

impl Default for Brain {
    fn default() -> Brain {
        Brain {
            input_gru: Arc::new(gru::MGRUInput::new_rand()),
            input_hiddens: OutputVector::new_random(),
            internal_grus: array_init::array_init(|_| Arc::new(gru::MGRU::new_rand())),
            internal_hiddens: array_init::array_init(|_| OutputVector::new_random()),
            signal: 0.0,
        }
    }
}

impl Brain {
    pub fn mutate(&mut self, lambda: f64) {
        if let Some(ngru) = self.input_gru.mutated(lambda) {
            self.input_gru = Arc::new(ngru);
        }
        for gru in &mut self.internal_grus {
            if let Some(ngru) = gru.mutated(lambda) {
                *gru = Arc::new(ngru);
            }
        }
    }

    pub fn apply(&mut self, inputs: &gru::InputVector) -> gru::OutputVector {
        let mut hiddens = self.input_gru.apply(inputs, &self.input_hiddens);
        self.input_hiddens = hiddens;
        for (ix, gru) in self.internal_grus.iter().enumerate() {
            hiddens = gru.apply(&hiddens, &self.internal_hiddens[ix]);
            self.internal_hiddens[ix] = hiddens;
        }
        hiddens
    }
}
