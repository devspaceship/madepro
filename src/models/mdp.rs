use rand::prelude::*;
use std::{fmt::Debug, hash::Hash};

/// # Collection
///
/// This trait represents shared functionality of `State` and `Action`.\
/// Namely the ability to iterate over the items.\
/// A default implementation is given for stochastic sampling.
pub trait Collection: Copy + Eq + Hash + Debug {
    type IntoIter: IntoIterator<Item = Self>;

    /// Returns an iterator over all items.
    fn get_all() -> Self::IntoIter;

    /// Returns a random item.
    fn get_random() -> Self {
        let mut rng = thread_rng();
        Self::get_all().into_iter().choose(&mut rng).unwrap()
    }
}

/// # State
///
/// This trait represents a state in the MDP.\
/// You can implement it on a custom struct for instance.
pub trait State: Collection {}

/// # Action
///
/// An action type must implement this trait.\
/// You can implement it on a custom enum for instance.
pub trait Action: Collection {}

/// # MDP
///
/// Your MDP should implement this trait.\
/// Afterwards you can use the solvers on your MDP.
pub trait MDP {
    type State: State;
    type Action: Action;

    /// Determines whether a state is terminal.
    fn is_state_terminal(&self, state: &Self::State) -> bool;

    /// Given a state and an action, returns the next state and reward.
    fn transition(&self, state: &Self::State, action: &Self::Action) -> (Self::State, f64);
}
