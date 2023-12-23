use std::vec;

use madepro::{
    config::Config,
    models::{Action, Model, Policy, State, StateValue, MDP},
};

const NO_OP_TRANSITION_REWARD: f64 = -1.0;
const END_TRANSITION_REWARD: f64 = 100.0;

static TOP_LEFT: GridworldState = GridworldState::new(0, 0);
static TOP_RIGHT: GridworldState = GridworldState::new(0, 1);
static BOTTOM_RIGHT: GridworldState = GridworldState::new(1, 1);

// State
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct GridworldState {
    i: usize,
    j: usize,
}

impl GridworldState {
    pub const fn new(i: usize, j: usize) -> Self {
        Self { i, j }
    }
}

impl Model for GridworldState {
    type IntoIter = vec::IntoIter<Self>;

    fn get_all() -> Self::IntoIter {
        vec![Self::new(0, 0), Self::new(0, 1), Self::new(1, 1)].into_iter()
    }
}

impl State for GridworldState {}

// Action
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum GridworldAction {
    Down,
    Left,
    Right,
    Up,
}

impl Model for GridworldAction {
    type IntoIter = vec::IntoIter<Self>;

    fn get_all() -> Self::IntoIter {
        vec![Self::Down, Self::Left, Self::Right, Self::Up].into_iter()
    }
}

impl Action for GridworldAction {}

// Policy
pub fn get_optimal_policy() -> Policy<GridworldState, GridworldAction> {
    let mut policy = Policy::new();
    policy.insert(&TOP_LEFT, &GridworldAction::Right);
    policy.insert(&TOP_RIGHT, &GridworldAction::Down);
    policy.insert(&BOTTOM_RIGHT, &GridworldAction::Up);
    policy
}

// Cell
#[derive(Debug, PartialEq)]
pub enum Cell {
    Air,
    Wall,
    End,
}

pub fn get_test_grid() -> Vec<Vec<Cell>> {
    vec![vec![Cell::Air, Cell::Air], vec![Cell::Wall, Cell::End]]
}

// Gridworld
pub struct Gridworld {
    cell_grid: Vec<Vec<Cell>>,
}

impl Gridworld {
    pub fn new(cell_grid: Vec<Vec<Cell>>) -> Self {
        Self { cell_grid }
    }
}

impl Gridworld {
    fn get_grid_size(&self) -> (usize, usize) {
        (self.cell_grid.len(), self.cell_grid[0].len())
    }
}

impl MDP for Gridworld {
    type State = GridworldState;
    type Action = GridworldAction;

    fn is_state_terminal(&self, state: &Self::State) -> bool {
        let cell = &self.cell_grid[state.i][state.j];
        *cell == Cell::End
    }

    fn transition(&self, state: &Self::State, action: &Self::Action) -> (Self::State, f64) {
        let cell = &self.cell_grid[state.i][state.j];

        // Edge cases
        // In theory the Cell::Wall case should never happen
        if (*cell) == Cell::End || (*cell) == Cell::Wall {
            return (*state, 0.0);
        }

        // Tentative position
        let (i, j) = (state.i as i32, state.j as i32);
        let (i_, j_) = match action {
            Self::Action::Up => (i - 1, j),
            Self::Action::Down => (i + 1, j),
            Self::Action::Left => (i, j - 1),
            Self::Action::Right => (i, j + 1),
        };

        // Check out of bounds
        let (n, m) = self.get_grid_size();
        let (n, m) = (n as i32, m as i32);
        if i_ < 0 || i_ >= n || j_ < 0 || j_ >= m {
            return (*state, NO_OP_TRANSITION_REWARD);
        }

        // Result
        let (i_, j_) = (i_ as usize, j_ as usize);
        let cell_ = &self.cell_grid[i_][j_];
        match cell_ {
            Cell::Air => (Self::State::new(i_, j_), NO_OP_TRANSITION_REWARD),
            Cell::Wall => (*state, NO_OP_TRANSITION_REWARD),
            Cell::End => (Self::State::new(i_, j_), END_TRANSITION_REWARD),
        }
    }
}

pub fn get_test_mdp() -> Gridworld {
    Gridworld::new(get_test_grid())
}

pub fn assert_policy_optimal(policy: &Policy<GridworldState, GridworldAction>) {
    assert_eq!(policy.get(&TOP_LEFT), &GridworldAction::Right);
    assert_eq!(policy.get(&TOP_RIGHT), &GridworldAction::Down);
}

pub fn assert_state_value_correct(state_value: &StateValue<GridworldState>) {
    assert_eq!(state_value.get(&TOP_LEFT), 96.0);
    assert_eq!(state_value.get(&TOP_RIGHT), 100.0);
    assert_eq!(state_value.get(&BOTTOM_RIGHT), 0.0);
}

pub fn get_test_config() -> Config {
    Config::new()
        .discount_factor(0.97)
        .iterations_before_improvement(None)
}

pub fn get_test_state_value() -> StateValue<GridworldState> {
    let mut state_value = StateValue::new();
    state_value.insert(&TOP_LEFT, 96.0);
    state_value.insert(&TOP_RIGHT, 100.0);
    state_value.insert(&BOTTOM_RIGHT, 0.0);
    state_value
}

// Tests
mod tests {
    use super::*;

    #[test]
    #[ignore = "meta test"]
    fn transition_to_boundaries() {
        let mdp = get_test_mdp();
        assert_eq!(
            mdp.transition(&TOP_LEFT, &GridworldAction::Left),
            (GridworldState::new(0, 0), NO_OP_TRANSITION_REWARD)
        );
    }

    #[test]
    #[ignore = "meta test"]
    fn transition_to_air() {
        let mdp = get_test_mdp();
        assert_eq!(
            mdp.transition(&TOP_LEFT, &GridworldAction::Right),
            (GridworldState::new(0, 1), NO_OP_TRANSITION_REWARD)
        );
    }

    #[test]
    #[ignore = "meta test"]
    fn transition_to_wall() {
        let mdp = get_test_mdp();
        assert_eq!(
            mdp.transition(&TOP_LEFT, &GridworldAction::Down),
            (GridworldState::new(0, 0), NO_OP_TRANSITION_REWARD)
        );
    }

    #[test]
    #[ignore = "meta test"]
    fn transition_to_end() {
        let mdp = get_test_mdp();
        assert_eq!(
            mdp.transition(&TOP_RIGHT, &GridworldAction::Down),
            (GridworldState::new(1, 1), END_TRANSITION_REWARD)
        );
    }

    #[test]
    #[ignore = "meta test"]
    fn transition_from_terminal() {
        let mdp = get_test_mdp();
        assert_eq!(
            mdp.transition(&BOTTOM_RIGHT, &GridworldAction::Up),
            (GridworldState::new(1, 1), 0.0)
        );
    }
}
