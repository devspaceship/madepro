use std::collections::HashMap;

use super::{
    mdp::{Action, State},
    Sampler,
};

/// # Policy
///
/// Represents a mapping from states to actions.\
/// Since it initializes all states with a random action,
/// it unwraps the `Option` returned by `HashMap::get`.
#[derive(Debug, PartialEq, Eq)]
pub struct Policy<S, A>(HashMap<S, A>)
where
    S: State,
    A: Action;

impl<S, A> Policy<S, A>
where
    S: State,
    A: Action,
{
    /// Creates a new policy with each state mapped to a random action.
    pub fn new(states: &Sampler<S>, actions: &Sampler<A>) -> Self {
        let mut map = HashMap::new();
        for state in states {
            map.insert(state.clone(), actions.get_random().clone());
        }
        Self(map)
    }

    /// Returns the action associated with the given state.
    pub fn get(&self, state: &S) -> &A {
        self.0.get(state).unwrap()
    }

    /// Inserts the given action for the given state.
    pub fn insert(&mut self, state: &S, action: &A) {
        self.0.insert(state.clone(), action.clone());
    }
}
