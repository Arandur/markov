use rand::distributions::{Weighted, WeightedChoice};
use rand::{Rand, Rng};

#[derive(Clone)]
struct MarkovState<T> {
    value: T,
    transitions: Vec<Weighted<T>>
}

impl<T> MarkovState<T> where T: Hash + Eq + Rand + Clone {
    pub fn next<R: Rng>(&self, rng: &mut R) -> T {
        let mut weights = self.transitions.clone();
        let wc = WeightedChoice::new(weights);
        wc.ind_sample(rng)
    }
}
