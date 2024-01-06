use std::hash::Hash;

use super::Sampler;

/// # Item
///
/// You have to implement this trait for your own state or action type.\
/// This trait is used to ensure that it can be used as a key in a HashMap.\
/// It also ensures that we can call `.clone()` on it.
pub trait Item: Eq + Hash + Clone {}

/// # State
///
/// You have to implement this trait for your own state type.
///
/// ## Example
///
/// ```
/// use madepro::models::{Item, State};
///
/// #[derive(PartialEq, Eq, Hash, Clone)]
/// struct MyState {
///     x: u32,
///     y: u32,
/// }
///
/// impl Item for MyState {}
/// impl State for MyState {}
/// ```
pub trait State: Item {}

/// # Action
///
/// You have to implement this trait for your own action type.\
/// It ensures that your action type can be used as a key in a HashMap.
///
/// ## Example
///
/// ```
/// use madepro::models::{Item, Action};
///
/// #[derive(PartialEq, Eq, Hash, Clone)]
/// enum MyAction {
///     Up,
///     Down,
///     Left,
///     Right,
/// }
///
/// impl Item for MyAction {}
/// impl Action for MyAction {}
/// ```
pub trait Action: Item {}

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
