use super::{InLen, OutLen};
use na;
use rand::{self, distributions::Poisson, Rng};

type InputMatrix = na::MatrixMN<f32, OutLen, InLen>;
type HiddenMatrix = na::MatrixMN<f32, OutLen, OutLen>;
pub type InputVector = na::MatrixMN<f32, InLen, na::dimension::U1>;
pub type OutputVector = na::MatrixMN<f32, OutLen, na::dimension::U1>;

#[inline]
fn mutate_lambda(slice: &mut [f32], lambda: f64) {
    let mut rng = rand::thread_rng();
    let times = rng.sample(Poisson::new(lambda));
    for _ in 0..times {
        *rng.choose_mut(slice)
            .expect("evomata12::sim::brain::gru::mutate_lambda(): mutated empty slice") =
            rng.gen::<f32>() * 2.0 - 1.0;
    }
}

#[derive(Clone)]
struct GRUNet {
    hidden_matrix: HiddenMatrix,
    input_matrix: InputMatrix,
    biases: OutputVector,
}

impl GRUNet {
    fn new_random() -> GRUNet {
        GRUNet {
            hidden_matrix: HiddenMatrix::new_random().map(|n| n * 2.0 - 1.0),
            input_matrix: InputMatrix::new_random().map(|n| n * 2.0 - 1.0),
            biases: OutputVector::new_random().map(|n| n * 2.0 - 1.0),
        }
    }

    fn apply_linear(&self, hiddens: &OutputVector, inputs: &InputVector) -> OutputVector {
        &self.hidden_matrix * hiddens + &self.input_matrix * inputs + &self.biases
    }

    fn apply_tanh(&self, hiddens: &OutputVector, inputs: &InputVector) -> OutputVector {
        self.apply_linear(hiddens, inputs).map(f32::tanh)
    }

    fn apply_sigmoid(&self, hiddens: &OutputVector, inputs: &InputVector) -> OutputVector {
        self.apply_linear(hiddens, inputs)
            .map(|n| (1.0 + (-n).ln()).recip())
    }

    /// Mutate each matrix element with a probability lambda
    fn mutate(&mut self, lambda: f64) {
        mutate_lambda(self.hidden_matrix.as_mut_slice(), lambda);
        mutate_lambda(self.input_matrix.as_mut_slice(), lambda);
        mutate_lambda(self.biases.as_mut_slice(), lambda);
    }
}

/// A Minimal Gated Recurrent Unit
#[derive(Clone)]
pub struct MGRU {
    forget_gate: GRUNet,
    output_gate: GRUNet,
    hiddens: OutputVector,
}

impl MGRU {
    pub fn new_rand() -> MGRU {
        MGRU {
            forget_gate: GRUNet::new_random(),
            output_gate: GRUNet::new_random(),
            hiddens: OutputVector::new_random(),
        }
    }

    pub fn apply(&mut self, inputs: &InputVector) -> OutputVector {
        // Compute forget coefficients.
        let f = self.forget_gate.apply_sigmoid(&self.hiddens, inputs);

        let remebered = f.zip_map(&self.hiddens, |f, h| f * h);

        self.hiddens = remebered
            + f.zip_map(&self.output_gate.apply_tanh(&remebered, inputs), |f, o| {
                (1.0 - f) * o
            });

        self.hiddens
    }

    pub fn mutate(&mut self, lambda: f64) {
        self.forget_gate.mutate(lambda);
        self.output_gate.mutate(lambda);
    }
}
