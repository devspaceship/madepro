mod utils;
use utils::{
    assert_policy_optimal, assert_state_value_correct, get_optimal_policy, get_test_config,
    get_test_mdp, get_test_state_value,
};

use madepro::{
    models::MDP,
    solvers::{
        dp::{policy_evaluation, policy_improvement, policy_iteration, value_iteration},
        td::{q_learning, sarsa},
    },
};

#[test]
fn test_policy_evaluation() {
    let mdp = get_test_mdp();
    let config = get_test_config();
    let policy = get_optimal_policy(mdp.get_states(), mdp.get_actions());
    let state_value = policy_evaluation(&mdp, &config, &policy, None);
    assert_state_value_correct(&state_value);
}

#[test]
fn test_policy_inference() {
    let mdp = get_test_mdp();
    let config = get_test_config();
    let state_value = get_test_state_value(mdp.get_states());
    let inferred_policy = policy_improvement(&mdp, &config, &state_value);
    assert_policy_optimal(&inferred_policy);
}

#[test]
fn test_policy_iteration() {
    let mdp = get_test_mdp();
    let config = get_test_config();
    let state_value = policy_iteration(&mdp, &config);
    let policy = policy_improvement(&mdp, &config, &state_value);
    assert_state_value_correct(&state_value);
    assert_policy_optimal(&policy);
}

#[test]
fn test_value_iteration() {
    let mdp = get_test_mdp();
    let config = get_test_config().iterations_before_improvement(Some(3));
    let state_value = value_iteration(&mdp, &config);
    let policy = policy_improvement(&mdp, &config, &state_value);
    assert_state_value_correct(&state_value);
    assert_policy_optimal(&policy);
}

#[test]
fn test_sarsa() {
    let mdp = get_test_mdp();
    let config = get_test_config();
    let action_value = sarsa(&mdp, &config);
    let policy = action_value.greedy_policy(mdp.get_states(), mdp.get_actions());
    assert_policy_optimal(&policy);
}

#[test]
fn test_q_learning() {
    let mdp = get_test_mdp();
    let config = get_test_config();
    let action_value = q_learning(&mdp, &config);
    let policy = action_value.greedy_policy(mdp.get_states(), mdp.get_actions());
    assert_policy_optimal(&policy);
}
