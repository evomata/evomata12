mod brain;

use self::brain::Brain;
use gs::Sim;

const MUTATE_LAMBDA: f32 = 0.01;

pub enum E12 {}

#[derive(Copy, Clone, Debug, PartialEq, Eq, EnumIterator)]
enum Direction {
    Up,
    Down,
    Left,
    Right,
}

impl Direction {
    fn get(self, cells: [[&Cell; 3]; 3]) -> &Cell {
        use self::Direction::*;
        match self {
            Up => cells[0][1],
            Down => cells[2][1],
            Left => cells[1][0],
            Right => cells[1][2],
        }
    }

    fn inv(self) -> Direction {
        use self::Direction::*;
        match self {
            Up => Down,
            Down => Up,
            Left => Right,
            Right => Left,
        }
    }
}

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
