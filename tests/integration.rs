mod utils;

use madepro::utils::{infer_policy, policy_evaluation};
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
