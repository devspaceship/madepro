use crate::defaults::{
    DISCOUNT_FACTOR, EXPLORATION_RATE, ITERATIONS_BEFORE_IMPROVEMENT, LEARNING_RATE, MAX_NUM_STEPS,
    NUM_EPISODES,
};

/// # Config
///
/// The `Config` struct contains the configuration for the different algorithms.
/// Currently there is only one struct for all algorithms.
/// In the future, there might be a better structuring of the configuration.
pub struct Config {
    pub discount_factor: f64,
    pub max_num_steps: u32,
    pub num_episodes: u32,
    pub learning_rate: f64,
    pub exploration_rate: f64,
    pub iterations_before_improvement: Option<u32>,
}

impl Config {
    /// Creates a new `Config` struct with the default values.
    pub fn new() -> Self {
        Self {
            discount_factor: DISCOUNT_FACTOR,
            max_num_steps: MAX_NUM_STEPS,
            num_episodes: NUM_EPISODES,
            learning_rate: LEARNING_RATE,
            exploration_rate: EXPLORATION_RATE,
            iterations_before_improvement: ITERATIONS_BEFORE_IMPROVEMENT,
        }
    }

    /// Sets the discount factor and returns the config.
    pub fn discount_factor(mut self, discount_factor: f64) -> Self {
        self.discount_factor = discount_factor;
        self
    }

    /// Sets the maximum number of steps per episode and returns the config.
    pub fn max_num_steps(mut self, max_num_steps: u32) -> Self {
        self.max_num_steps = max_num_steps;
        self
    }

    /// Sets the number of episodes and returns the config.
    pub fn num_episodes(mut self, num_episodes: u32) -> Self {
        self.num_episodes = num_episodes;
        self
    }

    /// Sets the learning rate and returns the config.
    pub fn learning_rate(mut self, learning_rate: f64) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    /// Sets the exploration rate and returns the config.
    pub fn exploration_rate(mut self, exploration_rate: f64) -> Self {
        self.exploration_rate = exploration_rate;
        self
    }

    /// Sets the number of iterations before improvement and returns the config.
    pub fn iterations_before_improvement(
        mut self,
        iterations_before_improvement: Option<u32>,
    ) -> Self {
        self.iterations_before_improvement = iterations_before_improvement;
        self
    }
}

impl Default for Config {
    fn default() -> Self {
        Self::new()
    }
}
