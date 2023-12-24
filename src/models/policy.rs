use std::collections::HashMap;

use super::{
    mdp::{Action, State},
    Sampler,
};

#[derive(Debug, PartialEq, Eq)]
pub struct Policy<'s, 'a, S, A>(HashMap<&'s S, &'a A>)
where
    S: State,
    A: Action;

impl<S, A> Policy<'_, '_, S, A>
where
    S: State,
    A: Action,
{
    pub fn new(states: &Sampler<S>, actions: &Sampler<A>) -> Self {
        let mut map = HashMap::new();
        for state in states {
            map.insert(state, actions.get_random());
        }
        Self(map)
    }

    pub fn get(&self, state: &S) -> &A {
        self.0.get(state).unwrap()
    }

    pub fn insert(&mut self, state: &S, action: &A) {
        self.0.insert(state, action);
    }
}
