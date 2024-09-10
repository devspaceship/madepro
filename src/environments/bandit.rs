/// # bandit
///
/// This module contains a test bandit model.
use rand::Rng;
use rand_distr::StandardNormal;

pub struct KArmedBandit {
    arm_values: Vec<f64>,
}

impl KArmedBandit {
    /// Create a new k-armed bandit
    pub fn new(k: i32) -> Self {
        let mut rng = rand::thread_rng();
        let mut arm_values = Vec::new();
        for _ in 0..k {
            let value = rng.sample(StandardNormal);
            arm_values.push(value);
        }
        KArmedBandit { arm_values }
    }

    /// Sample the value of an arm
    pub fn sample_arm(&self, arm: i32) -> f64 {
        self.arm_values[arm as usize]
            + rand::thread_rng().sample::<f64, StandardNormal>(StandardNormal)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bandit() {
        let bandit = KArmedBandit::new(10);
        assert_eq!(bandit.arm_values.len(), 10);
    }

    #[test]
    fn test_sample_arm() {
        let bandit = KArmedBandit::new(10);
        let value = bandit.sample_arm(0);
        assert!(value.is_finite());
    }
}
