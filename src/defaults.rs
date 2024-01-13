//! # defaults
//!
//! This module contains the default values for the `Config` struct.

/// The default discount factor.
pub const DISCOUNT_FACTOR: f64 = 0.97;

/// The default maximum number of steps per episode.
pub const MAX_NUM_STEPS: u32 = 1_000;

/// The default number of episodes.
pub const NUM_EPISODES: u32 = 500;

/// The default learning rate.
pub const LEARNING_RATE: f64 = 0.3;

/// The default exploration rate.
pub const EXPLORATION_RATE: f64 = 0.1;

/// The default number of iterations before improvement.
pub const ITERATIONS_BEFORE_IMPROVEMENT: Option<u32> = None;
