//! # madepro
//!
//! The `madepro` crate is a reinforcement learning library for solving Markov Decision Processes.
//!
//! ## Limitations
//!
//! This library is still in development and is not ready for production use.
//! Originally, I only created this library in order to refactor my blog post
//! on [Markov Decision Processes](https://devspaceship.com/posts/gridworld).
//!
//! However, I decided to probably continue working on it when I have the time.
//! In the future I would first like to make the library more generic
//! by making it able to solve stochastic processes and accept stochastic policies for instance
//! and then add more algorithms and environments.
//!
//! ## Example Usage
//!
//! ### Gridworld
//!
//! The following example shows how to use the library to solve the Gridworld environment.
//!
//! TODO

pub mod defaults;
pub mod environments;
pub mod errors;
pub mod models;
pub mod solvers;
