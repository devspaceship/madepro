//! # gridworld
//!
//! The Gridworld environment is a simple 2D grid with four actions: up, down, left, and right.
//! The environment consists of cells, each of which can be either empty or contain a wall or an end cell.
//! The agent's goal is to reach the end cell while avoiding walls.
//! Moving into a wall results in no change in position.
//! Moving into an empty cell or a wall results in a reward of -1.
//! Moving into the end cell results in a transition to a terminal state and a reward of 100.

mod env;
mod utils;

pub use env::*;
pub use utils::*;

const NO_OP_TRANSITION_REWARD: f64 = -1.0;
const END_TRANSITION_REWARD: f64 = 100.0;

static TOP_LEFT: GridworldState = GridworldState::new(0, 0);
static TOP_RIGHT: GridworldState = GridworldState::new(0, 1);
static BOTTOM_RIGHT: GridworldState = GridworldState::new(1, 1);
static LEFT: GridworldAction = GridworldAction::Left;
static RIGHT: GridworldAction = GridworldAction::Right;
static UP: GridworldAction = GridworldAction::Up;
static DOWN: GridworldAction = GridworldAction::Down;
