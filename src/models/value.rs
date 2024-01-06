use rand::prelude::*;

use crate::errors::NotFound;

use super::{Action, Policy, Sampler, State};

use std::collections::HashMap;

#[derive(Debug)]
pub struct StateValue<S>(HashMap<S, f64>)
where
    S: State;

impl<S> StateValue<S>
where
    S: State,
{
    pub fn new(states: &Sampler<S>) -> Self {
        let mut map = HashMap::new();
        for state in states {
            map.insert(state.clone(), 0.0);
        }
        Self(map)
    }

    pub fn get(&self, state: &S) -> f64 {
        *self
            .0
            .get(state)
            .unwrap_or_else(|| panic!("{}", NotFound::StateInStateValue))
    }

    pub fn insert(&mut self, state: &S, value: f64) {
        self.0.insert(state.clone(), value);
    }
}

#[derive(Debug)]
pub struct StateActionValue<A>(HashMap<A, f64>)
where
    A: Action;

impl<A> StateActionValue<A>
where
    A: Action,
{
    pub fn new(actions: &Sampler<A>) -> Self {
        let mut map = HashMap::new();
        for action in actions {
            map.insert(action.clone(), 0.0);
        }
        Self(map)
    }

    pub fn get(&self, action: &A) -> f64 {
        *self
            .0
            .get(action)
            .unwrap_or_else(|| panic!("{}", NotFound::ActionInStateActionValue))
    }

    pub fn insert(&mut self, action: &A, value: f64) {
        self.0.insert(action.clone(), value);
    }

    pub fn greedy(&self) -> &A {
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
            // unwrap is safe because the map is not empty
            .unwrap();
        best_action
    }

    pub fn epsilon_greedy<'a>(&'a self, actions: &'a Sampler<A>, epsilon: f64) -> &A {
        if random::<f64>() < epsilon {
            actions.get_random()
        } else {
            self.greedy()
        }
    }
}

impl<A> Clone for StateActionValue<A>
where
    A: Action,
{
    fn clone(&self) -> Self {
        let mut map = HashMap::new();
        for (action, value) in &self.0 {
            map.insert(action.clone(), *value);
        }
        Self(map)
    }
}

#[derive(Debug)]
pub struct ActionValue<S, A>(HashMap<S, StateActionValue<A>>)
where
    S: State,
    A: Action;

impl<S, A> ActionValue<S, A>
where
    S: State,
    A: Action,
{
    pub fn new(states: &Sampler<S>, actions: &Sampler<A>) -> Self {
        let mut map = HashMap::new();
        for state in states {
            map.insert(state.clone(), StateActionValue::new(actions));
        }
        Self(map)
    }

    pub fn get(&self, state: &S, action: &A) -> f64 {
        self.0
            .get(state)
            .unwrap_or_else(|| panic!("{}", NotFound::StateInActionValue))
            .get(action)
    }

    pub fn insert(&mut self, state: &S, action: &A, value: f64) {
        self.0
            .get_mut(state)
            .unwrap_or_else(|| panic!("{}", NotFound::StateInActionValue))
            .insert(action, value);
    }

    pub fn greedy(&self, state: &S) -> &A {
        self.0
            .get(state)
            .unwrap_or_else(|| panic!("{}", NotFound::StateInActionValue))
            .greedy()
    }

    pub fn epsilon_greedy<'a>(&'a self, actions: &'a Sampler<A>, state: &S, epsilon: f64) -> &A {
        self.0
            .get(state)
            .unwrap_or_else(|| panic!("{}", NotFound::StateInActionValue))
            .epsilon_greedy(actions, epsilon)
    }

    pub fn greedy_policy(&self, states: &Sampler<S>, actions: &Sampler<A>) -> Policy<S, A> {
        let mut policy = Policy::new(states, actions);
        for state in states {
            policy.insert(state, self.greedy(state));
        }
        policy
    }
}

impl<S, A> Clone for ActionValue<S, A>
where
    S: State,
    A: Action,
{
    fn clone(&self) -> Self {
        let mut map = HashMap::new();
        for (state, state_action_value) in &self.0 {
            map.insert(state.clone(), state_action_value.clone());
        }
        Self(map)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(PartialEq, Eq, Hash, Clone, Debug)]
    struct TestState(usize);
    impl State for TestState {}

    #[derive(PartialEq, Eq, Hash, Clone, Debug)]
    struct TestAction(usize);
    impl Action for TestAction {}

    fn get_states() -> Sampler<TestState> {
        Sampler::new(vec![TestState(0), TestState(1)])
    }

    fn get_actions() -> Sampler<TestAction> {
        Sampler::new(vec![TestAction(0), TestAction(1)])
    }

    fn get_state_action_value() -> StateActionValue<TestAction> {
        let mut state_action_value = StateActionValue::new(&get_actions());
        state_action_value.insert(&TestAction(0), 0.0);
        state_action_value.insert(&TestAction(1), 1.0);
        state_action_value
    }

    fn get_action_value() -> ActionValue<TestState, TestAction> {
        let mut action_value = ActionValue::new(&get_states(), &get_actions());
        action_value.insert(&TestState(0), &TestAction(0), 0.0);
        action_value.insert(&TestState(0), &TestAction(1), 1.0);
        action_value.insert(&TestState(1), &TestAction(0), 2.0);
        action_value.insert(&TestState(1), &TestAction(1), 1.0);
        action_value
    }

    #[test]
    fn state_action_value_greedy() {
        assert_eq!(get_state_action_value().greedy(), &TestAction(1));
    }

    #[test]
    fn state_action_value_epsilon_greedy() {
        assert_eq!(
            get_state_action_value().epsilon_greedy(&get_actions(), 0.0),
            &TestAction(1)
        );
    }

    #[test]
    fn action_value_greedy_policy() {
        let policy = get_action_value().greedy_policy(&get_states(), &get_actions());
        assert_eq!(policy.get(&TestState(0)), &TestAction(1));
        assert_eq!(policy.get(&TestState(1)), &TestAction(0));
    }
}
