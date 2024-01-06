use std::hash::Hash;

use super::Sampler;

/// # State
///
/// You have to implement this trait for your own state type.
///
/// ## Example
///
/// ```
/// use madepro::models::State;
///
/// #[derive(PartialEq, Eq, Hash, Clone)]
/// struct MyState {
///     x: u32,
///     y: u32,
/// }
///
/// impl State for MyState {}
/// ```
pub trait State: Eq + Hash + Clone {}

/// # Action
///
/// You have to implement this trait for your own action type.
///
/// ## Example
///
/// ```
/// use madepro::models::Action;
///
/// #[derive(PartialEq, Eq, Hash, Clone)]
/// enum MyAction {
///     Up,
///     Down,
///     Left,
///     Right,
/// }
///
/// impl Action for MyAction {}
/// ```
pub trait Action: Eq + Hash + Clone {}

/// # Markov Decision Process
///
/// You have to implement this trait for your own MDP.\
/// You should allocate the state and action samplers
/// in the constructor of your MDP.\
/// You can use the `Sampler` struct for this purpose.
pub trait MDP {
    type State: State;
    type Action: Action;

    /// Returns a reference to the state sampler.
    fn get_states(&self) -> &Sampler<Self::State>;

    /// Returns a reference to the action sampler.
    fn get_actions(&self) -> &Sampler<Self::Action>;

    /// Determines whether a state is terminal.
    fn is_state_terminal(&self, state: &Self::State) -> bool;

    /// Given a state and an action, returns the next state and reward.
    fn transition(&self, state: &Self::State, action: &Self::Action) -> (Self::State, f64);
}
