mod brain;

use sigmoid;

use noisy_float::prelude::*;

use self::brain::Brain;
use gs::{neumann::{Direction as Dir, Neighbors},
         Neighborhood,
         Sim};
use rand::random;

const MUTATE_LAMBDA: f64 = 0.01;
pub const SPAWN_FOOD: usize = 8192;
const FOOD_RATE_FACTOR: f32 = 0.2;
const SPAWN_PROBABILITY: f32 = FOOD_RATE_FACTOR / SPAWN_FOOD as f32;

pub enum E12 {}

/// The bool is if they want to divide.
fn get_choice(a: &brain::OutputVector) -> (Option<(Dir, bool)>, f32) {
    use self::Dir::*;
    let dir_bool = a.as_slice()
        .iter()
        .take(8)
        .cloned()
        .map(n32)
        .zip(&[
            (Right, false),
            (Up, false),
            (Left, false),
            (Down, false),
            (Right, true),
            (Up, true),
            (Left, true),
            (Down, true),
        ])
        .filter(|(v, _)| v.is_sign_positive())
        .max_by_key(|&(v, _)| v)
        .map(|(_, &dir)| dir);

    (dir_bool, a[8])
}

impl<'a> Sim<'a> for E12 {
    type Cell = Cell;
    type Diff = Diff;
    type Move = Move;

    type Neighbors = Neighbors<&'a Cell>;
    type MoveNeighbors = Neighbors<Move>;

    fn step(cell: &Cell, neighbors: Self::Neighbors) -> (Diff, Self::MoveNeighbors) {
        use std::iter::once;
        let mut taken = false;

        let choice = cell.brain.as_ref().and_then(|brain| {
            let inputs = brain::InputVector::from_iterator(
                neighbors
                    .iter()
                    .flat_map(|n| {
                        once(sigmoid(n.food as f32)).chain(
                            n.brain
                                .as_ref()
                                .map(|b| once(1.0).chain(once(b.signal)))
                                .unwrap_or_else(|| once(0.0).chain(once(0.0))),
                        )
                    })
                    .chain(once(sigmoid(cell.food as f32))),
            );
            // A promise is made here not to look at the brain of any other cell elsewhere.
            let brain = unsafe { &mut *(brain as *const Brain as *mut Brain) };
            let outputs = brain.apply(&inputs);
            let (choice, signal) = get_choice(&outputs);
            // TODO: This is bad; do this in update.
            brain.signal = signal;
            choice
        });

        let taken_food = choice
            .map(|cd| {
                if cd.1 {
                    cell.food / 2
                } else {
                    cell.food
                }
            })
            .unwrap_or(0);

        let moves = Neighbors::new(|dir| {
            choice
                .and_then(|cd| {
                    if cd.0 == dir {
                        // Food is taken with the brain.
                        taken = !cd.1;
                        Some(Move {
                            food: taken_food,
                            brain: cell.brain.as_ref().cloned(),
                        })
                    } else {
                        None
                    }
                })
                .unwrap_or_default()
        });
        let diff = Diff {
            consume: taken_food,
        };
        (diff, moves)
    }

    fn update(cell: &mut Cell, diff: Diff, moves: Self::MoveNeighbors) {
        // Handle food reduction.
        cell.food = cell.food
            .saturating_sub(diff.consume + if cell.brain.is_some() { 1 } else { 0 });
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

        if cell.brain.is_none() && random::<f32>() < SPAWN_PROBABILITY {
            cell.brain = Some(Brain::default());
            cell.food += SPAWN_FOOD;
        }
    }
}

#[derive(Clone, Default)]
pub struct Cell {
    pub food: usize,
    pub brain: Option<Brain>,
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
