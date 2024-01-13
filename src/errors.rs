//! # errors
//!
//! This module contains the error messages for the library.
//! It currently only contains the `NotFound` enum.

use std::fmt;

/// The `NotFound` enum contains the error messages
/// for when a state or action is not found as a key in a map.
pub enum NotFound {
    StateInPolicy,
    StateInStateValue,
    StateInActionValue,
    ActionInStateActionValue,
}

impl fmt::Display for NotFound {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let message = match self {
            NotFound::StateInPolicy => "state not found in policy",
            NotFound::StateInStateValue => "state not found in state value",
            NotFound::StateInActionValue => "state not found in action value",
            NotFound::ActionInStateActionValue => "action not found in state action value",
        };
        write!(f, "{}", message)
    }
}
