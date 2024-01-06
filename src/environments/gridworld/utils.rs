use super::{Cell, Gridworld, GridworldAction, GridworldState};
use crate::models::{Config, Policy, Sampler, StateValue};
use std::vec;

pub static TOP_LEFT: GridworldState = GridworldState::new(0, 0);
pub static TOP_RIGHT: GridworldState = GridworldState::new(0, 1);
pub static BOTTOM_RIGHT: GridworldState = GridworldState::new(1, 1);

pub static LEFT: GridworldAction = GridworldAction::Left;
pub static RIGHT: GridworldAction = GridworldAction::Right;
pub static UP: GridworldAction = GridworldAction::Up;
pub static DOWN: GridworldAction = GridworldAction::Down;

pub fn get_states() -> Vec<GridworldState> {
    vec![
        GridworldState::new(0, 0),
        GridworldState::new(0, 1),
        GridworldState::new(1, 1),
    ]
}

pub fn get_actions() -> Vec<GridworldAction> {
    vec![
        GridworldAction::Down,
        GridworldAction::Left,
        GridworldAction::Right,
        GridworldAction::Up,
    ]
}

pub fn get_test_mdp() -> Gridworld {
    Gridworld::new(
        vec![vec![Cell::Air, Cell::Air], vec![Cell::Wall, Cell::End]],
        get_states(),
        get_actions(),
    )
}

pub fn get_test_config() -> Config {
    Config::new()
        .discount_factor(0.97)
        .iterations_before_improvement(None)
        .exploration_rate(0.1)
}

pub fn get_optimal_policy(
    states: &Sampler<GridworldState>,
    actions: &Sampler<GridworldAction>,
) -> Policy<GridworldState, GridworldAction> {
    let mut policy = Policy::new(states, actions);
    policy.insert(&TOP_LEFT, &RIGHT);
    policy.insert(&TOP_RIGHT, &DOWN);
    policy.insert(&BOTTOM_RIGHT, &UP);
    policy
}

pub fn get_test_state_value(states: &Sampler<GridworldState>) -> StateValue<GridworldState> {
    let mut state_value = StateValue::new(states);
    state_value.insert(&TOP_LEFT, 96.0);
    state_value.insert(&TOP_RIGHT, 100.0);
    state_value.insert(&BOTTOM_RIGHT, 0.0);
    state_value
}

pub fn assert_policy_optimal(policy: &Policy<GridworldState, GridworldAction>) {
    assert_eq!(policy.get(&TOP_LEFT), &RIGHT);
    assert_eq!(policy.get(&TOP_RIGHT), &DOWN);
}

pub fn assert_state_value_correct(state_value: &StateValue<GridworldState>) {
    assert_eq!(state_value.get(&TOP_LEFT), 96.0);
    assert_eq!(state_value.get(&TOP_RIGHT), 100.0);
    assert_eq!(state_value.get(&BOTTOM_RIGHT), 0.0);
}