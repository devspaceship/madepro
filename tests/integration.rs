mod utils;

use madepro::{
    solvers::{policy_iteration, q_learning, sarsa, value_iteration},
    utils::{infer_policy, policy_evaluation},
};
use utils::{
    assert_policy_optimal, assert_state_value_correct, get_optimal_policy, get_test_config,
    get_test_mdp, get_test_state_value,
};

#[test]
fn test_policy_evaluation() {
    let mdp = get_test_mdp();
    let config = get_test_config();
    let policy = get_optimal_policy();
    let state_value = policy_evaluation(&mdp, &config, &policy, None);
    assert_state_value_correct(&state_value);
}

#[test]
fn test_policy_inference() {
    let mdp = get_test_mdp();
    let config = get_test_config();
    let state_value = get_test_state_value();
    let inferred_policy = infer_policy(&mdp, &config, &state_value);
    assert_policy_optimal(&inferred_policy);
}

#[test]
fn test_policy_iteration() {
    let mdp = get_test_mdp();
    let config = get_test_config();
    let (state_value, policy) = policy_iteration(&mdp, &config);
    assert_state_value_correct(&state_value);
    assert_policy_optimal(&policy);
}

#[test]
fn test_value_iteration() {
    let mdp = get_test_mdp();
    let config = get_test_config().iterations_before_improvement(Some(3));
    let (state_value, policy) = value_iteration(&mdp, &config);
    assert_state_value_correct(&state_value);
    assert_policy_optimal(&policy);
}

#[test]
fn test_sarsa() {
    let mdp = get_test_mdp();
    let config = get_test_config();
    let action_value = sarsa(&mdp, &config);
    let policy = action_value.greedy_policy();
    assert_policy_optimal(&policy);
}

#[test]
fn test_q_learning() {
    let mdp = get_test_mdp();
    let config = get_test_config();
    let action_value = q_learning(&mdp, &config);
    let policy = action_value.greedy_policy();
    assert_policy_optimal(&policy);
}
