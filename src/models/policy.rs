use super::mdp::{Action, State};

use std::collections::HashMap;

#[derive(Debug, PartialEq, Eq)]
pub struct Policy<S, A>(HashMap<S, A>)
where
    S: State,
    A: Action;

impl<S: State, A: Action> Policy<S, A> {
    pub fn new() -> Self {
        let mut map = HashMap::new();
        for state in S::get_all() {
            map.insert(state, A::get_random());
        }
        Self(map)
    }

    pub fn get(&self, state: &S) -> &A {
        self.0.get(state).unwrap()
    }

    pub fn insert(&mut self, state: &S, action: &A) {
        self.0.insert(*state, *action);
    }
}

impl<S: State, A: Action> Default for Policy<S, A> {
    fn default() -> Self {
        Self::new()
    }
}
