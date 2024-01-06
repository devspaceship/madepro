use crate::models::{Action, Item, Sampler, State, MDP};

const NO_OP_TRANSITION_REWARD: f64 = -1.0;
const END_TRANSITION_REWARD: f64 = 100.0;

// State
#[derive(PartialEq, Eq, Hash, Debug, Clone)]
pub struct GridworldState {
    i: usize,
    j: usize,
}

impl GridworldState {
    pub const fn new(i: usize, j: usize) -> Self {
        Self { i, j }
    }
}

impl Item for GridworldState {}
impl State for GridworldState {}

// Action
#[derive(PartialEq, Eq, Hash, Debug, Clone)]
pub enum GridworldAction {
    Down,
    Left,
    Right,
    Up,
}

impl Item for GridworldAction {}
impl Action for GridworldAction {}

// Cell
#[derive(Debug, PartialEq)]
pub enum Cell {
    Air,
    Wall,
    End,
}

// Gridworld
pub struct Gridworld {
    cell_grid: Vec<Vec<Cell>>,
    states: Sampler<GridworldState>,
    actions: Sampler<GridworldAction>,
}

impl Gridworld {
    pub fn new(
        cell_grid: Vec<Vec<Cell>>,
        states: Vec<GridworldState>,
        actions: Vec<GridworldAction>,
    ) -> Self {
        Self {
            cell_grid,
            states: states.into(),
            actions: actions.into(),
        }
    }

    fn get_grid_size(&self) -> (usize, usize) {
        (self.cell_grid.len(), self.cell_grid[0].len())
    }
}

impl MDP for Gridworld {
    type State = GridworldState;
    type Action = GridworldAction;

    fn get_states(&self) -> &Sampler<Self::State> {
        &self.states
    }

    fn get_actions(&self) -> &Sampler<Self::Action> {
        &self.actions
    }

    fn is_state_terminal(&self, state: &Self::State) -> bool {
        let cell = &self.cell_grid[state.i][state.j];
        *cell == Cell::End
    }

    fn transition(&self, state: &Self::State, action: &Self::Action) -> (Self::State, f64) {
        let cell = &self.cell_grid[state.i][state.j];

        // Edge cases
        // In theory the Cell::Wall case should never happen
        if (*cell) == Cell::End || (*cell) == Cell::Wall {
            return (state.clone(), 0.0);
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
            return (state.clone(), NO_OP_TRANSITION_REWARD);
        }

        // Result
        let (i_, j_) = (i_ as usize, j_ as usize);
        let cell_ = &self.cell_grid[i_][j_];
        match cell_ {
            Cell::Air => (Self::State::new(i_, j_), NO_OP_TRANSITION_REWARD),
            Cell::Wall => (state.clone(), NO_OP_TRANSITION_REWARD),
            Cell::End => (Self::State::new(i_, j_), END_TRANSITION_REWARD),
        }
    }
}

// Tests
#[cfg(test)]
mod tests {
    use super::*;
    use crate::environments::gridworld::{
        get_test_mdp, BOTTOM_RIGHT, DOWN, LEFT, RIGHT, TOP_LEFT, TOP_RIGHT, UP,
    };

    #[test]
    fn transition_to_boundaries() {
        let mdp = get_test_mdp();
        assert_eq!(
            mdp.transition(&TOP_LEFT, &LEFT),
            (TOP_LEFT.clone(), NO_OP_TRANSITION_REWARD)
        );
    }

    #[test]
    fn transition_to_air() {
        let mdp = get_test_mdp();
        assert_eq!(
            mdp.transition(&TOP_LEFT, &RIGHT),
            (TOP_RIGHT.clone(), NO_OP_TRANSITION_REWARD)
        );
    }

    #[test]
    fn transition_to_wall() {
        let mdp = get_test_mdp();
        assert_eq!(
            mdp.transition(&TOP_LEFT, &DOWN),
            (TOP_LEFT.clone(), NO_OP_TRANSITION_REWARD)
        );
    }

    #[test]
    fn transition_to_end() {
        let mdp = get_test_mdp();
        assert_eq!(
            mdp.transition(&TOP_RIGHT, &DOWN),
            (BOTTOM_RIGHT.clone(), END_TRANSITION_REWARD)
        );
    }

    #[test]
    fn transition_from_terminal() {
        let mdp = get_test_mdp();
        assert_eq!(
            mdp.transition(&BOTTOM_RIGHT, &UP),
            (BOTTOM_RIGHT.clone(), 0.0)
        );
    }
}
