use std::collections::HashMap;

use super::{
    mdp::{Action, State},
    policy::Policy,
};

#[derive(Debug)]
pub struct StateValue<S>(HashMap<S, f64>)
where
    S: State;

impl<S: State> StateValue<S> {
    pub fn new() -> Self {
        let mut map = HashMap::new();
        for state in S::get_all() {
            map.insert(state, 0.0);
        }
        Self(map)
    }

    pub fn get(&self, state: &S) -> f64 {
        *self.0.get(state).unwrap()
    }

    pub fn insert(&mut self, state: &S, value: f64) {
        self.0.insert(*state, value);
    }
}

impl<S: State> Default for StateValue<S> {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug)]
pub struct StateActionValue<A>(HashMap<A, f64>)
where
    A: Action;

impl<A: Action> StateActionValue<A> {
    pub fn new() -> Self {
        let mut map = HashMap::new();
        for action in A::get_all() {
            map.insert(action, 0.0);
        }
        Self(map)
    }

    pub fn get(&self, action: &A) -> f64 {
        *self.0.get(action).unwrap()
    }

    pub fn insert(&mut self, action: &A, value: f64) {
        self.0.insert(*action, value);
    }

    pub fn greedy(&self) -> A {
        let (best_action, _) = self
            .0
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

    pub fn epsilon_greedy(&self, epsilon: f64) -> A {
        if rand::random::<f64>() < epsilon {
            A::get_random()
        } else {
            self.greedy()
        }
    }
}

impl<A: Action> Default for StateActionValue<A> {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug)]
pub struct ActionValue<S, A>(HashMap<S, StateActionValue<A>>)
where
    S: State,
    A: Action;

impl<S: State, A: Action> ActionValue<S, A> {
    pub fn new() -> Self {
        let mut map = HashMap::new();
        for state in S::get_all() {
            map.insert(state, StateActionValue::new());
        }
        Self(map)
    }

    pub fn get(&self, state: &S, action: &A) -> f64 {
        self.0.get(state).unwrap().get(action)
    }

    pub fn insert(&mut self, state: &S, action: &A, value: f64) {
        self.0.get_mut(state).unwrap().insert(action, value);
    }

    pub fn greedy(&self, state: &S) -> A {
        self.0.get(state).unwrap().greedy()
    }

    pub fn epsilon_greedy(&self, state: &S, epsilon: f64) -> A {
        self.0.get(state).unwrap().epsilon_greedy(epsilon)
    }

    pub fn greedy_policy(&self) -> Policy<S, A> {
        let mut policy = Policy::new();
        for state in S::get_all() {
            policy.insert(&state, &self.greedy(&state));
        }
        policy
    }
}

impl<S: State, A: Action> Default for ActionValue<S, A> {
    fn default() -> Self {
        Self::new()
    }
}
