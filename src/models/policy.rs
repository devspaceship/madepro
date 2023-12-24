use std::collections::HashMap;

use super::{
    mdp::{Action, State},
    Sampler,
};

#[derive(Debug, PartialEq, Eq)]
pub struct Policy<'a, S, A>(HashMap<&'a S, &'a A>)
where
    S: State,
    A: Action;

impl<'a, S, A> Policy<'a, S, A>
where
    S: State,
    A: Action,
{
    pub fn new(states: &'a Sampler<S>, actions: &'a Sampler<A>) -> Self {
        let mut map = HashMap::new();
        for state in states {
            map.insert(state, actions.get_random());
        }
        Self(map)
    }

    pub fn get(&self, state: &S) -> &A {
        self.0.get(state).unwrap()
    }

    pub fn insert(&mut self, state: &'a S, action: &'a A) {
        self.0.insert(state, action);
    }
}
