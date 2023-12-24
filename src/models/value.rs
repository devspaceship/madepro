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
pub struct StateActionValue<'a, A>(HashMap<&'a A, f64>)
where
    A: Action;

impl<'a, A> StateActionValue<'a, A>
where
    A: Action,
{
    pub fn new(actions: &'a Sampler<A>) -> Self {
        let mut map = HashMap::new();
        for action in actions {
            map.insert(action, 0.0);
        }
        Self(map)
    }

    pub fn get(&self, action: &A) -> f64 {
        *self.0.get(action).unwrap()
    }

    pub fn insert(&mut self, action: &'a A, value: f64) {
        self.0.insert(action, value);
    }

    pub fn greedy(&self, _actions: &'a Sampler<A>) -> &'a A {
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
        best_action
    }

    pub fn epsilon_greedy(&self, actions: &'a Sampler<A>, epsilon: f64) -> &'a A {
        if random::<f64>() < epsilon {
            actions.get_random()
        } else {
            self.greedy(actions)
        }
    }
}

impl<A> Clone for StateActionValue<'_, A>
where
    A: Action,
{
    fn clone(&self) -> Self {
        let mut map = HashMap::new();
        for (&action, &value) in &self.0 {
            map.insert(action, value);
        }
        Self(map)
    }
}

#[derive(Debug)]
pub struct ActionValue<'a, S, A>(HashMap<&'a S, StateActionValue<'a, A>>)
where
    S: State,
    A: Action;

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
        Self(map)
    }

    pub fn get(&self, state: &S, action: &A) -> f64 {
        self.0.get(state).unwrap().get(action)
    }

    pub fn insert(&mut self, state: &'a S, action: &'a A, value: f64) {
        self.0.get_mut(state).unwrap().insert(action, value);
    }

    pub fn greedy(&self, actions: &'a Sampler<A>, state: &S) -> &'a A {
        self.0.get(state).unwrap().greedy(actions)
    }

    pub fn epsilon_greedy(&self, actions: &'a Sampler<A>, state: &S, epsilon: f64) -> &'a A {
        self.0.get(state).unwrap().epsilon_greedy(actions, epsilon)
    }

    pub fn greedy_policy(
        &self,
        states: &'a Sampler<S>,
        actions: &'a Sampler<A>,
    ) -> Policy<'a, S, A> {
        let mut policy = Policy::new(states, actions);
        for state in states {
            policy.insert(state, self.greedy(actions, state));
        }
        policy
    }
}

impl<S, A> Clone for ActionValue<'_, S, A>
where
    S: State,
    A: Action,
{
    fn clone(&self) -> Self {
        let mut map = HashMap::new();
        for (&state, state_action_value) in &self.0 {
            map.insert(state, state_action_value.clone());
        }
        Self(map)
    }
}
