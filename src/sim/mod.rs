mod brain;

pub fn sigmoid(n: f32) -> f32 {
    (1.0 + (-n).exp()).recip()
}

use noisy_float::prelude::*;

use self::brain::Brain;
use gs::{
    neumann::{Direction as Dir, Neighbors}, Neighborhood, Sim,
};
use rand::random;

const MUTATE_LAMBDA: f64 = 0.00001;
pub const SPAWN_FOOD: usize = 512;
const FOOD_RATE_FACTOR: f32 = 5.0;
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

        let choices = cell.brain.as_ref().map(|brain| {
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
            get_choice(&outputs)
        });

        let choice = choices.and_then(|t| t.0);
        let signal = choices.map(|t| t.1);

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
                            signal: signal
                                .expect("evomata12::sim::step(): moved but no signal present"),
                        })
                    } else {
                        None
                    }
                })
                .unwrap_or_default()
        });
        let diff = Diff {
            consume: taken_food,
            moved: taken,
            signal: signal,
        };
        (diff, moves)
    }

    fn update(cell: &mut Cell, diff: Diff, moves: Self::MoveNeighbors) {
        // Handle food reduction from diff.
        cell.food = cell.food.saturating_sub(diff.consume);

        // Handle taking the brain.
        if diff.moved {
            cell.brain.take();
        }

        // Handle signal for still cells (from diff).
        if let Some(signal) = diff.signal {
            if let Some(ref mut brain) = cell.brain {
                brain.signal = signal;
            }
        }

        // Handle brain movement.
        let mut brain_moves = moves.clone().iter().filter(|m| m.brain.is_some());
        if brain_moves.clone().count() == 1 {
            let m = brain_moves.next().unwrap();
            cell.brain = m.brain;
            cell.brain.as_mut().unwrap().signal = m.signal;
        }

        // Handle food movement.
        cell.food += moves.iter().map(|m| m.food).sum::<usize>();

        // Handle food reduction from existing.
        cell.food = cell.food.saturating_sub(1);

        // Handle death.
        if cell.food == 0 {
            cell.brain.take();
        }

        // Handle mutation.
        if let Some(ref mut brain) = cell.brain {
            brain.mutate(MUTATE_LAMBDA);
        }

        // Handle spawning.
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
    signal: f32,
}

#[derive(Default, Clone, Debug)]
pub struct Diff {
    consume: usize,
    moved: bool,
    signal: Option<f32>,
}
