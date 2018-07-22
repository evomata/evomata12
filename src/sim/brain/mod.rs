mod gru;

pub use self::gru::{InputVector, OutputVector};

use array_init;
use na;
use std::sync::Arc;

mod arc_serde {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::sync::Arc;
    pub fn serialize<T: Serialize, S: Serializer>(t: &Arc<T>, s: S) -> Result<S::Ok, S::Error> {
        t.serialize(s)
    }

    pub fn deserialize<'de, T: Deserialize<'de>, D: Deserializer<'de>>(
        d: D,
    ) -> Result<Arc<T>, D::Error> {
        Ok(Arc::new(T::deserialize(d)?))
    }
}

pub type OutLen = na::dimension::U16;
// 1 + 8 + 8 + 8
pub type InLen = na::dimension::U25;

// Number of extra layers
const EXTRA_LAYERS: usize = 1;

#[derive(Clone, Serialize, Deserialize)]
pub struct Brain {
    #[serde(with = "arc_serde")]
    input_gru: Arc<gru::MGRUInput>,
    input_hiddens: OutputVector,
    #[serde(with = "arc_serde")]
    internal_grus: Arc<[gru::MGRU; EXTRA_LAYERS]>,
    internal_hiddens: [OutputVector; EXTRA_LAYERS],
    pub signal: f32,
}

impl Default for Brain {
    fn default() -> Brain {
        Brain {
            input_gru: Arc::new(gru::MGRUInput::new_rand()),
            input_hiddens: OutputVector::new_random(),
            internal_grus: Arc::new(array_init::array_init(|_| gru::MGRU::new_rand())),
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
        for i in 0..self.internal_grus.len() {
            if let Some(ngru) = self.internal_grus[i].mutated(lambda) {
                let mut new_grus = (*self.internal_grus).clone();
                new_grus[i] = ngru;
                self.internal_grus = Arc::new(new_grus);
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
