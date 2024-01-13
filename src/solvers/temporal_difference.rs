//! # temporal_difference
//!
//! The `temporal_difference` module contains the implementations of the temporal difference algorithms.

use crate::models::{ActionValue, Config, MDP};

fn sarsa_q_learning<M>(
    mdp: &M,
    config: &Config,
    q_learning: bool,
) -> ActionValue<M::State, M::Action>
where
    M: MDP,
{
    let states = mdp.get_states();
    let actions = mdp.get_actions();
    let mut action_value = ActionValue::new(states, actions);
    for _ in 0..config.num_episodes {
        let mut state = states.get_random().clone();
        let mut action = action_value
            .epsilon_greedy(actions, &state, config.exploration_rate)
            .clone();
        for _ in 0..config.max_num_steps {
            let (next_state, reward) = mdp.transition(&state, &action);
            let next_action = action_value
                .epsilon_greedy(actions, &next_state, config.exploration_rate)
                .clone();
            // update action value
            let current = action_value.get(&state, &action);
            let q_value = if q_learning {
                action_value.get(&next_state, action_value.greedy(&next_state))
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

/// # SARSA
///
/// This function implements the SARSA algorithm.
/// It works by using the Bellman equation to iteratively update the action value.
/// The algorithm stops after the given number of episodes.
/// An episode is a sequence of state-action pairs that ends in a terminal state.
/// The number of steps per episode is limited by the `max_num_steps` parameter in the config.
/// The algorithm uses the epsilon-greedy policy to select actions.
pub fn sarsa<M>(mdp: &M, config: &Config) -> ActionValue<M::State, M::Action>
where
    M: MDP,
{
    sarsa_q_learning(mdp, config, false)
}

/// # Q-Learning
///
/// This function implements the Q-Learning algorithm.
/// It works by using the Bellman equation to iteratively update the action value.
/// The algorithm stops after the given number of episodes.
/// An episode is a sequence of state-action pairs that ends in a terminal state.
/// The number of steps per episode is limited by the `max_num_steps` parameter in the config.
/// The algorithm uses the epsilon-greedy policy to select actions.
/// Unlike SARSA, Q-Learning uses the greedy policy to select the action
/// from which the value is used in the update rule.
pub fn q_learning<M>(mdp: &M, config: &Config) -> ActionValue<M::State, M::Action>
where
    M: MDP,
{
    sarsa_q_learning(mdp, config, true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::environments::gridworld::{assert_policy_optimal, get_gridworld, get_test_config};

    #[test]
    fn test_sarsa() {
        let mdp = get_gridworld();
        let config = get_test_config();
        let action_value = sarsa(&mdp, &config);
        let policy = action_value.greedy_policy(mdp.get_states(), mdp.get_actions());
        assert_policy_optimal(&policy);
    }

    #[test]
    fn test_q_learning() {
        let mdp = get_gridworld();
        let config = get_test_config();
        let action_value = q_learning(&mdp, &config);
        let policy = action_value.greedy_policy(mdp.get_states(), mdp.get_actions());
        assert_policy_optimal(&policy);
    }
}
