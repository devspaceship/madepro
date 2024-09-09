use crate::models::{Action, Sampler};

/// # Bandit
///
/// You have to implement this trait for your own bandit problem.
/// You should allocate the action sampler in the constructor of your bandit.
/// You can use the [`Sampler`] struct for this purpose.
pub trait Bandit {
    type Action: Action;

    /// Returns a reference to the action sampler.
    fn get_actions(&self) -> &Sampler<Self::Action>;

    /// Given an action, returns the reward.
    fn reward(&self, action: &Self::Action) -> f64;
}
