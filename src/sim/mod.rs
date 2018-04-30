mod brain;

use af::{Array, Dim4};
use noisy_float::prelude::*;

use self::brain::Brain;
use gs::{neumann::{Direction as Dir, Neighbors},
         Neighborhood,
         Sim};
use rand::{thread_rng, Rng};

const MUTATE_LAMBDA: f32 = 0.01;
const SPAWN_PROBABILITY: f32 = 0.0001;
const SPAWN_FOOD: usize = 5;

pub enum E12 {}

fn sigmoid(n: usize) -> f32 {
    (1.0 + (-(n as f32)).exp()).recip()
}

fn get_choice(a: &Array) -> Option<Dir> {
    use self::Dir::*;
    let mut host = [0f32; 4];
    a.host(&mut host);
    host.iter()
        .cloned()
        .map(n32)
        .zip(&[Right, Up, Left, Down])
        .filter(|(v, _)| v.is_sign_positive())
        .max_by_key(|&(v, _)| v)
        .map(|(_, &dir)| dir)
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
            let inputs = neighbors
                .iter()
                .flat_map(|n| {
                    once(sigmoid(n.food)).chain(once(n.brain.as_ref().map(|_| 1.0).unwrap_or(0.0)))
                })
                .collect::<Vec<_>>();
            // A promise is made here not to look at the brain of any other cell elsewhere.
            let brain = unsafe { &mut *(brain as *const Brain as *mut Brain) };
            let outputs = brain.apply(&Array::new(
                inputs.as_slice(),
                Dim4::new(&[inputs.len() as u64, 1, 1, 1]),
            ));
            get_choice(&outputs)
        });

        let moves = Neighbors::new(|dir| {
            if choice.map(|cd| cd == dir).unwrap_or(false) {
                // Food is taken with the brain.
                taken = true;
                Move {
                    food: cell.food,
                    brain: cell.brain.as_ref().cloned(),
                }
            } else {
                Move::default()
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
