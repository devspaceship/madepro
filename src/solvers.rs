use crate::{
    config::Config,
    models::{ActionValue, Model, Policy, StateValue, MDP},
    utils::{infer_policy, policy_evaluation},
};

fn policy_value_iteration<M>(
    mdp: &M,
    config: &Config,
) -> (StateValue<M::State>, Policy<M::State, M::Action>)
where
    M: MDP,
{
    let mut state_value = StateValue::new();
    let mut policy = Policy::new();
    loop {
        state_value = policy_evaluation(mdp, config, &policy, Some(state_value));
        let new_policy = infer_policy(mdp, config, &state_value);
        if new_policy == policy {
            break;
        }
        policy = new_policy;
    }
    (state_value, policy)
}

pub fn policy_iteration<M>(
    mdp: &M,
    config: &Config,
) -> (StateValue<M::State>, Policy<M::State, M::Action>)
where
    M: MDP,
{
    assert!(
        config.iterations_before_improvement.is_none(),
        "Iterations before improvement must be None for Policy Iteration"
    );
    policy_value_iteration(mdp, config)
}

pub fn value_iteration<M>(
    mdp: &M,
    config: &Config,
) -> (StateValue<M::State>, Policy<M::State, M::Action>)
where
    M: MDP,
{
    assert!(
        config.iterations_before_improvement.is_some_and(|n| n > 0),
        "Iterations before improvement must be Some(u32) and > 0 for Value Iteration"
    );
    policy_value_iteration(mdp, config)
}

fn sarsa_q_learning<M>(
    mdp: &M,
    config: &Config,
    q_learning: bool,
) -> ActionValue<M::State, M::Action>
where
    M: MDP,
{
    let mut action_value: ActionValue<M::State, M::Action> = ActionValue::new();
    for _ in 1..=config.num_episodes {
        let mut state = M::State::get_random();
        let mut action = action_value.epsilon_greedy(&state, config.exploration_rate);
        for _ in 1..=config.max_num_steps {
            let (next_state, reward) = mdp.transition(&state, &action);
            let next_action = action_value.epsilon_greedy(&next_state, config.exploration_rate);
            // update action value
            let current = action_value.get(&state, &action);
            let q_value = if q_learning {
                action_value.get(&next_state, &action_value.greedy(&next_state))
            } else {
                action_value.get(&next_state, &next_action)
            };
            let target = reward + config.discount_factor * q_value;
            action_value.insert(
                &state,
                &action,
                current + config.learning_rate * (target - current),
            );
            state = next_state;
            action = next_action;
            if mdp.is_state_terminal(&state) {
                break;
            }
        }
    }
    action_value
}

pub fn sarsa<M>(mdp: &M, config: &Config) -> ActionValue<M::State, M::Action>
where
    M: MDP,
{
    sarsa_q_learning(mdp, config, false)
}

pub fn q_learning<M>(mdp: &M, config: &Config) -> ActionValue<M::State, M::Action>
where
    M: MDP,
{
    sarsa_q_learning(mdp, config, true)
}
