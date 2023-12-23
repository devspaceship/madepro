use crate::{
    config::Config,
    models::{Model, Policy, StateValue, MDP},
};

/// # Policy Evaluation
///
/// This function implements the policy evaluation algorithm.\
/// It works by using the Bellman equation to iteratively update the state values.
/// The algorithm stops when the state values converge.
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
    let mut state_value = initial_state_value.unwrap_or(StateValue::new());
    let mut iteration = 0;
    loop {
        iteration += 1;
        let mut delta: f64 = 0.0;
        for state in M::State::get_all() {
            let action = policy.get(&state);
            let (next_state, reward) = mdp.transition(&state, &action);
            let next_state_value = state_value.get(&next_state);
            let new_state_value = reward + config.discount_factor * next_state_value;
            delta = delta.max((new_state_value - state_value.get(&state)).abs());
            state_value.insert(&state, new_state_value);
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

/// # Policy Inference
///
/// Given a state value function, this function infers the optimal policy.
pub fn infer_policy<M>(
    mdp: &M,
    config: &Config,
    state_value: &StateValue<M::State>,
) -> Policy<M::State, M::Action>
where
    M: MDP,
{
    let mut policy = Policy::new();
    for state in M::State::get_all() {
        let mut best_action = None;
        let mut best_value = None;
        for action in M::Action::get_all() {
            let (next_state, reward) = mdp.transition(&state, &action);
            let value = reward + config.discount_factor * state_value.get(&next_state);
            if best_value.is_none() || value > best_value.unwrap() {
                best_value = Some(value);
                best_action = Some(action);
            }
        }
        if let Some(best_action) = best_action {
            policy.insert(&state, &best_action);
        }
    }
    policy
}
