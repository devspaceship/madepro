use crate::models::{Config, Policy, StateValue, MDP};

/// # Policy Evaluation
///
/// This function implements the policy evaluation algorithm.
/// It works by using the Bellman equation to iteratively update the state value.
/// The algorithm stops when the state value converge.
/// If the `iterations_before_improvement` parameter is set,
/// the algorithm will stop early after the given number of iterations.
pub fn policy_evaluation<M>(
    mdp: &M,
    config: &Config,
    policy: &Policy<M::State, M::Action>,
    initial_state_value: Option<StateValue<M::State>>,
) -> StateValue<M::State>
where
    M: MDP,
{
    let states = mdp.get_states();
    let mut state_value = initial_state_value.unwrap_or(StateValue::new(states));
    let mut iteration = 0;
    loop {
        iteration += 1;
        let mut delta: f64 = 0.0;
        for state in states {
            let action = policy.get(state);
            let (next_state, reward) = mdp.transition(state, action);
            let next_state_value = state_value.get(&next_state);
            let new_state_value = reward + config.discount_factor * next_state_value;
            delta = delta.max((new_state_value - state_value.get(state)).abs());
            state_value.insert(state, new_state_value);
        }
        if delta < 1e-5
            || config
                .iterations_before_improvement
                .is_some_and(|n| iteration >= n)
        {
            break;
        }
    }
    state_value
}

/// # Policy Improvement
///
/// Given an MDP, a discount factor and a state value,
/// this function computes the optimal policy.
pub fn policy_improvement<M>(
    mdp: &M,
    config: &Config,
    state_value: &StateValue<M::State>,
) -> Policy<M::State, M::Action>
where
    M: MDP,
{
    let states = mdp.get_states();
    let actions = mdp.get_actions();
    let mut policy = Policy::new(states, actions);
    for state in states {
        let mut best_action = None;
        let mut best_value = None;
        for action in actions {
            let (next_state, reward) = mdp.transition(state, action);
            let value = reward + config.discount_factor * state_value.get(&next_state);
            if best_value.is_none() || value > best_value.unwrap() {
                best_value = Some(value);
                best_action = Some(action);
            }
        }
        // unwrap is safe because actions is not empty
        policy.insert(state, best_action.unwrap());
    }
    policy
}

fn policy_value_iteration<M>(mdp: &M, config: &Config) -> StateValue<M::State>
where
    M: MDP,
{
    let states = mdp.get_states();
    let actions = mdp.get_actions();
    let mut state_value = StateValue::new(states);
    let mut policy = Policy::new(states, actions);
    loop {
        state_value = policy_evaluation(mdp, config, &policy, Some(state_value));
        let new_policy = policy_improvement(mdp, config, &state_value);
        if new_policy == policy {
            break;
        }
        policy = new_policy;
    }
    state_value
}

pub fn policy_iteration<M>(mdp: &M, config: &Config) -> StateValue<M::State>
where
    M: MDP,
{
    assert!(
        config.iterations_before_improvement.is_none(),
        "Iterations before improvement must be None for Policy Iteration"
    );
    policy_value_iteration(mdp, config)
}

pub fn value_iteration<M>(mdp: &M, config: &Config) -> StateValue<M::State>
where
    M: MDP,
{
    assert!(
        config.iterations_before_improvement.is_some_and(|n| n > 0),
        "Iterations before improvement must be Some(u32) and > 0 for Value Iteration"
    );
    policy_value_iteration(mdp, config)
}
