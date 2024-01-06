use std::slice;

use rand::prelude::*;

#[derive(Debug)]
pub struct Sampler<T>(Vec<T>);

impl<T> Sampler<T> {
    pub fn new(items: Vec<T>) -> Self {
        assert!(!items.is_empty(), "sampler must contain at least one item.");
        Self(items)
    }

    pub fn get_random(&self) -> &T {
        let mut rng = thread_rng();
        // unwrap is safe because sampler is not empty
        self.0.choose(&mut rng).unwrap()
    }

    pub fn iter(&self) -> slice::Iter<'_, T> {
        self.0.iter()
    }
}

impl<'a, T> IntoIterator for &'a Sampler<T> {
    type Item = &'a T;
    type IntoIter = slice::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<T> From<Vec<T>> for Sampler<T> {
    fn from(items: Vec<T>) -> Self {
        Self::new(items)
    }
}
