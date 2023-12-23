use rand::prelude::*;
use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;

/// # Model
///
/// This trait represents a model in the MDP.\
/// `State` and `Action` extend this trait.
pub trait Model: Copy + Eq + Hash + Debug {
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
pub trait State: Model {}

/// # Action
///
/// An action type must implement this trait.\
/// You can implement it on a custom enum for instance.
pub trait Action: Model {}

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
        self.0.get_mut(&state).unwrap().insert(action, value);
    }

    pub fn greedy(&self, state: &S) -> A {
        self.0.get(state).unwrap().greedy()
    }

    pub fn epsilon_greedy(&self, state: &S, epsilon: f64) -> A {
        self.0.get(state).unwrap().epsilon_greedy(epsilon)
    }
}

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
