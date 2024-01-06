//! # Errors
//!
//! This module contains the default error messages for the library.

use std::fmt;

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
