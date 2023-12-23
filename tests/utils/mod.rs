use madepro::{
    config::Config,
    models::{policy::Policy, value::StateValue},
};
use std::vec;

mod gridworld;
use gridworld::*;

pub fn get_test_mdp() -> Gridworld {
    Gridworld::new(vec![
        vec![Cell::Air, Cell::Air],
        vec![Cell::Wall, Cell::End],
    ])
}

pub fn get_test_config() -> Config {
    Config::new()
        .discount_factor(0.97)
        .iterations_before_improvement(None)
        .exploration_rate(0.1)
}

pub fn get_optimal_policy() -> Policy<GridworldState, GridworldAction> {
    let mut policy = Policy::new();
    policy.insert(&TOP_LEFT, &RIGHT);
    policy.insert(&TOP_RIGHT, &DOWN);
    policy.insert(&BOTTOM_RIGHT, &UP);
    policy
}

pub fn get_test_state_value() -> StateValue<GridworldState> {
    let mut state_value = StateValue::new();
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
