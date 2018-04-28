mod gru;

#[derive(Clone)]
pub struct Brain {
    gru: gru::GRU,
}

impl Default for Brain {
    fn default() -> Brain {
        Brain {
            gru: gru::GRU::new_rand(4, 4),
        }
    }
}

impl Brain {
    pub fn mutate(&mut self, lambda: f32) {
        self.gru.mutate(lambda);
    }
}
