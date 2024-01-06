use super::{
    Cell, Gridworld, GridworldAction, GridworldState, END_TRANSITION_REWARD,
    NO_OP_TRANSITION_REWARD,
};
use crate::models::{Config, Policy, Sampler, StateValue};
use std::vec;

const DISCOUNT_FACTOR: f64 = 0.97;
const EXPLORATION_RATE: f64 = 0.1;
const BOTTOM_RIGHT_VALUE: f64 = 0.0;
const TOP_RIGHT_VALUE: f64 = END_TRANSITION_REWARD;
const TOP_LEFT_VALUE: f64 = NO_OP_TRANSITION_REWARD + DISCOUNT_FACTOR * TOP_RIGHT_VALUE;

pub static TOP_LEFT: GridworldState = GridworldState::new(0, 0);
pub static TOP_RIGHT: GridworldState = GridworldState::new(0, 1);
pub static BOTTOM_RIGHT: GridworldState = GridworldState::new(1, 1);

pub static LEFT: GridworldAction = GridworldAction::Left;
pub static RIGHT: GridworldAction = GridworldAction::Right;
pub static UP: GridworldAction = GridworldAction::Up;
pub static DOWN: GridworldAction = GridworldAction::Down;

pub fn get_states() -> Vec<GridworldState> {
    vec![TOP_LEFT.clone(), TOP_RIGHT.clone(), BOTTOM_RIGHT.clone()]
}

pub fn get_actions() -> Vec<GridworldAction> {
    vec![DOWN.clone(), LEFT.clone(), RIGHT.clone(), UP.clone()]
}

pub fn get_gridworld() -> Gridworld {
    Gridworld::new(
        vec![vec![Cell::Air, Cell::Air], vec![Cell::Wall, Cell::End]],
        get_states(),
        get_actions(),
    )
}

pub fn get_test_config() -> Config {
    Config::new()
        .discount_factor(DISCOUNT_FACTOR)
        .iterations_before_improvement(None)
        .exploration_rate(EXPLORATION_RATE)
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
    state_value.insert(&TOP_LEFT, TOP_LEFT_VALUE);
    state_value.insert(&TOP_RIGHT, TOP_RIGHT_VALUE);
    state_value.insert(&BOTTOM_RIGHT, BOTTOM_RIGHT_VALUE);
    state_value
}

pub fn assert_policy_optimal(policy: &Policy<GridworldState, GridworldAction>) {
    assert_eq!(policy.get(&TOP_LEFT), &RIGHT);
    assert_eq!(policy.get(&TOP_RIGHT), &DOWN);
}

pub fn assert_state_value_correct(state_value: &StateValue<GridworldState>) {
    assert_eq!(state_value.get(&TOP_LEFT), TOP_LEFT_VALUE);
    assert_eq!(state_value.get(&TOP_RIGHT), TOP_RIGHT_VALUE);
    assert_eq!(state_value.get(&BOTTOM_RIGHT), BOTTOM_RIGHT_VALUE);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::MDP;

    #[test]
    fn optimal_policy() {
        let mdp = get_gridworld();
        let policy = get_optimal_policy(mdp.get_states(), mdp.get_actions());
        assert_policy_optimal(&policy);
    }

    #[test]
    fn state_value() {
        let mdp = get_gridworld();
        let state_value = get_test_state_value(mdp.get_states());
        assert_state_value_correct(&state_value);
    }
}
