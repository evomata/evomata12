mod gru;

use af::Array;

pub const BRAIN_SIZE: usize = 64;

#[derive(Clone)]
pub struct Brain {
    gru: gru::GRU,
}

impl Default for Brain {
    fn default() -> Brain {
        Brain {
            gru: gru::GRU::new_rand(17, BRAIN_SIZE as u64),
        }
    }
}

impl Brain {
    pub fn mutate(&mut self, lambda: f32) {
        self.gru.mutate(lambda);
    }

    pub fn apply(&mut self, inputs: &Array) -> Array {
        self.gru.apply(inputs)
    }
}
