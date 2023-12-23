// #[cfg(test)]
// mod tests {
// use super::*;
// use crate::{
//     test_utils::{get_test_grid, is_policy_optimal},
//     utils::action_value_grid_to_policy_grid,
// };

// #[test]
// fn test_policy_iteration() {
//     let test_grid = get_test_grid();
//     let (_state_value_grid, policy_grid) = policy_value_iteration(test_grid, None, None);
//     assert!(is_policy_optimal(&policy_grid));
// }

// #[test]
// fn test_value_iteration() {
//     let test_grid = get_test_grid();
//     let (_state_value_grid, policy_grid) = policy_value_iteration(test_grid, None, Some(5));
//     assert!(is_policy_optimal(&policy_grid));
// }

// #[test]
// fn test_sarsa() {
//     let test_grid = get_test_grid();
//     let action_value_grid = sarsa_q_learning(test_grid, false, None, None, None, None, None);
//     let policy_grid = action_value_grid_to_policy_grid(&action_value_grid);
//     assert_eq!(policy_grid[0][0], Action::Right);
//     assert_eq!(policy_grid[0][1], Action::Down);
// }

// #[test]
// fn test_q_learning() {
//     let test_grid = get_test_grid();
//     let action_value_grid = sarsa_q_learning(test_grid, true, None, None, None, None, None);
//     let policy_grid = action_value_grid_to_policy_grid(&action_value_grid);
//     assert_eq!(policy_grid[0][0], Action::Right);
//     assert_eq!(policy_grid[0][1], Action::Down);
// }
// }
