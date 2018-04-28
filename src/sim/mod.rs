mod brain;

use self::brain::Brain;
use gs::Sim;

pub enum E12 {}

impl Sim for E12 {
    type Cell = Cell;
    type Diff = Diff;

    fn step(cells: [[&Cell; 3]; 3]) -> Diff {
        Diff {}
    }

    fn update(cell: &mut Cell, diff: Diff) {}
}

#[derive(Clone, Default)]
pub struct Cell {
    brain: Option<Brain>,
}

#[derive(Default, Clone, Debug)]
pub struct Diff {}