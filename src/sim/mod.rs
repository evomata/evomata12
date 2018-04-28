mod brain;

use self::brain::Brain;
use gs::Sim;

const MUTATE_LAMBDA: f32 = 0.01;

pub enum E12 {}

impl Sim for E12 {
    type Cell = Cell;
    type Diff = Diff;

    fn step(cells: [[&Cell; 3]; 3]) -> Diff {
        Diff {}
    }

    fn update(cell: &mut Cell, diff: Diff) {
        // Handle mutation.
        if let Some(ref mut brain) = cell.brain {
            brain.mutate(MUTATE_LAMBDA);
        }
    }
}

#[derive(Clone, Default)]
pub struct Cell {
    brain: Option<Brain>,
}

#[derive(Default, Clone, Debug)]
pub struct Diff {}
