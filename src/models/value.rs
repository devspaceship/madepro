use rand::prelude::*;

use super::{
    mdp::{Action, State},
    policy::Policy,
    Sampler,
};

use std::collections::HashMap;

#[derive(Debug)]
pub struct StateValue<'a, S>(HashMap<&'a S, f64>)
where
    S: State;

impl<'a, S> StateValue<'a, S>
where
    S: State,
{
    pub fn new(states: &'a Sampler<S>) -> Self {
        let mut map = HashMap::new();
        for state in states {
            map.insert(state, 0.0);
        }
        Self(map)
    }

    pub fn get(&self, state: &S) -> f64 {
        *self.0.get(state).unwrap()
    }

    pub fn insert(&mut self, state: &'a S, value: f64) {
        self.0.insert(state, value);
    }
}

#[derive(Debug)]
pub struct StateActionValue<'a, A>
where
    A: Action,
{
    actions: &'a Sampler<A>,
    map: HashMap<&'a A, f64>,
}

impl<'a, A> StateActionValue<'a, A>
where
    A: Action,
{
    pub fn new(actions: &'a Sampler<A>) -> Self {
        let mut map = HashMap::new();
        for action in actions {
            map.insert(action, 0.0);
        }
        Self { actions, map }
    }

    pub fn get(&self, action: &A) -> f64 {
        *self.map.get(action).unwrap()
    }

    pub fn insert(&mut self, action: &'a A, value: f64) {
        self.map.insert(action, value);
    }

    pub fn greedy(&self) -> &A {
        let (best_action, _) = self
            .map
            .iter()
            .reduce(|(best_action, best_value), (action, value)| {
                if value > best_value {
                    (action, value)
                } else {
                    (best_action, best_value)
                }
            })
            .unwrap();
        *best_action
    }

    pub fn epsilon_greedy(&self, epsilon: f64) -> &A {
        if random::<f64>() < epsilon {
            self.actions.get_random()
        } else {
            self.greedy()
        }
    }
}

#[derive(Debug)]
pub struct ActionValue<'a, S, A>
where
    S: State,
    A: Action,
{
    states: &'a Sampler<S>,
    actions: &'a Sampler<A>,
    map: HashMap<&'a S, StateActionValue<'a, A>>,
}

impl<'a, S, A> ActionValue<'a, S, A>
where
    S: State,
    A: Action,
{
    pub fn new(states: &'a Sampler<S>, actions: &'a Sampler<A>) -> Self {
        let mut map = HashMap::new();
        for state in states {
            map.insert(state, StateActionValue::new(actions));
        }
        Self {
            states,
            actions,
            map,
        }
    }

    pub fn get(&self, state: &S, action: &A) -> f64 {
        self.map.get(state).unwrap().get(action)
    }

    pub fn insert(&mut self, state: &'a S, action: &'a A, value: f64) {
        self.map.get_mut(state).unwrap().insert(action, value);
    }

    pub fn greedy(&self, state: &S) -> &A {
        self.map.get(state).unwrap().greedy()
    }

    pub fn epsilon_greedy(&self, state: &S, epsilon: f64) -> &A {
        self.map.get(state).unwrap().epsilon_greedy(epsilon)
    }

    pub fn greedy_policy(&self) -> Policy<'_, S, A> {
        let mut policy = Policy::new(self.states, self.actions);
        for state in self.states {
            policy.insert(state, self.greedy(&state));
        }
        policy
    }
}
