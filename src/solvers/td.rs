use crate::{
    config::Config,
    models::{
        mdp::{Collection, MDP},
        value::ActionValue,
    },
};

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
