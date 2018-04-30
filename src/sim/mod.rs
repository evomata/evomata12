mod brain;

use self::brain::Brain;
use gs::{neumann::{Direction as Dir, Neighbors},
         Neighborhood,
         Sim};
use rand::{thread_rng, Rng};

const MUTATE_LAMBDA: f32 = 0.01;
const SPAWN_PROBABILITY: f32 = 0.0001;
const SPAWN_FOOD: usize = 5;

pub enum E12 {}

impl<'a> Sim<'a> for E12 {
    type Cell = Cell;
    type Diff = Diff;
    type Move = Move;

    type Neighbors = Neighbors<&'a Cell>;
    type MoveNeighbors = Neighbors<Move>;

    fn step(cell: &Cell, neighbors: Self::Neighbors) -> (Diff, Self::MoveNeighbors) {
        let mut taken = false;
        let moves = Neighbors::new(|dir| {
            if dir != Dir::Right || neighbors[dir].brain.is_some() {
                Move::default()
            } else {
                taken = true;
                Move {
                    food: cell.food,
                    brain: cell.brain.as_ref().cloned(),
                }
            }
        });
        let diff = Diff {
            consume: if taken { cell.food } else { 0 },
        };
        (diff, moves)
    }

    fn update(cell: &mut Cell, diff: Diff, moves: Self::MoveNeighbors) {
        // Handle food reduction.
        if diff.consume != 0 {
            cell.food -= diff.consume;
        }
        // Handle death.
        if cell.food == 0 {
            cell.brain.take();
        }
        // Handle mutation.
        if let Some(ref mut brain) = cell.brain {
            brain.mutate(MUTATE_LAMBDA);
        }
        // Handle brain movement.
        let mut brain_moves = moves.clone().iter().filter(|m| m.brain.is_some());
        if brain_moves.clone().count() == 1 {
            cell.brain = brain_moves.next().unwrap().brain;
        }

        // Handle food movement.
        cell.food += moves.iter().map(|m| m.food).sum::<usize>();

        if cell.brain.is_none() && thread_rng().next_f32() < SPAWN_PROBABILITY {
            cell.brain = Some(Brain::default());
            cell.food += SPAWN_FOOD;
        }
    }
}

#[derive(Clone, Default)]
pub struct Cell {
    pub food: usize,
    brain: Option<Brain>,
}

#[derive(Clone, Default)]
pub struct Move {
    food: usize,
    brain: Option<Brain>,
}

#[derive(Default, Clone, Debug)]
pub struct Diff {
    consume: usize,
}
