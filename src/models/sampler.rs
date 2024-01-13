use std::slice;

use rand::prelude::*;

/// # Sampler
///
/// Represents a collection of items from which you can sample.
/// You can get a random item from the sampler using the `get_random` method.
/// You can also iterate over the items in the sampler.
#[derive(Debug)]
pub struct Sampler<T>(Vec<T>);

impl<T> Sampler<T> {
    /// Creates a new sampler with the specified items.
    pub fn new(items: Vec<T>) -> Self {
        assert!(!items.is_empty(), "sampler must contain at least one item.");
        Self(items)
    }

    /// Returns a reference to a random item in the sampler.
    pub fn get_random(&self) -> &T {
        let mut rng = thread_rng();
        // unwrap is safe because sampler is not empty
        self.0.choose(&mut rng).unwrap()
    }

    /// Returns an iterator over references to the items in the sampler.
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
