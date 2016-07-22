#![warn(missing_docs)]

//! An implementation of Markov Chains in Rust.
//!
//! A Markov Chain is a weighted directed graph. We call the nodes of the graph
//! "states", and the edges "transitions". A given state of the Markov Chain
//! transitions stochastically to another state according to the weights of its
//! out-transitions.
//!
//! The objectives of this implementation are ease of use and generic
//! usability, with performance being an important but secondary concern

extern crate rustc_serialize;
extern crate rand;

use std::collections::HashMap;
use std::fmt;
use std::hash::Hash;
use std::iter::Iterator;
use std::mem;
use rand::Rng;
use rand::distributions::{Weighted, WeightedChoice, IndependentSample};
use rustc_serialize::{Encodable, Decodable, Encoder, Decoder};

/// Defines the restrictions on a MarkovIdentifier.
///
/// A MarkovIdentifier serves as a unique identifier for a given MarkovState.
/// Because this crate's implementation of Markov chains uses a HashMap
/// internally, MarkovIdentifiers must implement Eq and Hash. And because we
/// use MarkovIdentifiers in so many places, we require that they also
/// implement Clone.
///
/// (NB: Clone is implemented for &T for all T.)
pub trait MarkovIdentifier: Clone + Eq + Hash {}
impl<T> MarkovIdentifier for T where T: Clone + Eq + Hash {}

/// A single state of the MarkovChain.
///
/// A MarkovState consists of the following:
/// * a MarkovIdentifier which uniquely identifies the state's position in the
/// MarkovChain,
/// * a weighted set of MarkovIdentifiers, signifying the probabilities that
/// the state will transition to any other state in the MarkovChain,
/// * a value.
///
/// Values need not be unique in the MarkovChain. It is the values which will
/// be returned by MarkovChain's iterator.
pub struct MarkovState<I, T>
    where I: MarkovIdentifier
{
    identifier: I,
    transitions: HashMap<I, u32>,
    /// The state's value.
    pub value: T,
}

impl<I, T> MarkovState<I, T>
    where I: MarkovIdentifier
{
    /// Creates a new MarkovState.
    ///
    /// `transitions` is a HashMap of MarkovIdentifiers to unsigned integers,
    /// which are the weights of each transition. For example:
    ///
    /// ```rust
    /// use std::collections::HashMap;
    /// use markov::MarkovState;
    ///
    /// # fn main() {
    /// let mut transitions = HashMap::new();
    /// transitions.insert(0, 1);
    /// transitions.insert(1, 5);
    /// transitions.insert(2, 4);
    ///
    /// let state = MarkovState::new(0, transitions, 'a');
    /// # }
    /// ```
    ///
    /// This function has a 10% chance of printing "0", a 50% chance of
    /// printing "1", and a 40% chance of printing "2". The weights do not
    /// need to sum to any particular value; only the ratios are important.
    ///
    /// Note that MarkovStates can be constructed and used distinct from a
    /// MarkovChain, but the utility of this is severely limited.
    pub fn new(identifier: I, transitions: HashMap<I, u32>, value: T) -> MarkovState<I, T> {
        MarkovState {
            identifier: identifier,
            transitions: transitions,
            value: value,
        }
    }

    /// Selects a transition from this state.
    ///
    /// This function takes as parameter a mutable reference to a source of
    /// randomness which implements rand::Rng; that source determines which
    /// transition will be chosen.
    ///
    /// See MarkovState::new() for an example of use. Although this function
    /// may be called independently of any MarkovChain, it is intended for use
    /// by MarkovChain.
    ///
    /// # Panics
    ///
    /// Panics if the weights of the transitions sum to 0, or if said sum
    /// exceeds the capacity of a u32.
    // TODO: Check for panic conditions in new() instead of checking them here;
    // maybe do a debug_assert!() here.
    pub fn next<R: Rng>(&self, rng: &mut R) -> Option<I> {
        if self.transitions.is_empty() {
            None
        } else {
            let mut items: Vec<_> = self.transitions
                .clone()
                .into_iter()
                .map(|(k, v)| {
                    Weighted {
                        weight: v,
                        item: k,
                    }
                })
                .collect();
            let wc = WeightedChoice::new(&mut items);
            Some(wc.ind_sample(&mut *rng))
        }
    }
}

impl<I, T> Clone for MarkovState<I, T>
    where I: MarkovIdentifier,
          T: Clone
{
    fn clone(&self) -> Self {
        MarkovState {
            identifier: self.identifier.clone(),
            transitions: self.transitions.clone(),
            value: self.value.clone(),
        }
    }
}

impl<I, T> fmt::Debug for MarkovState<I, T>
    where I: MarkovIdentifier + fmt::Debug,
          T: fmt::Debug
{
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt.debug_struct("MarkovState")
            .field("identifier", &self.identifier)
            .field("transitions", &self.transitions)
            .field("value", &self.value)
            .finish()
    }
}

impl<I, T> PartialEq for MarkovState<I, T>
    where I: MarkovIdentifier,
          T: PartialEq
{
    fn eq(&self, other: &Self) -> bool {
        self.identifier == other.identifier && self.transitions == other.transitions &&
        self.value == other.value
    }
}

impl<I, T> Eq for MarkovState<I, T>
    where I: MarkovIdentifier,
          T: Eq
{
}

impl<I, T> Encodable for MarkovState<I, T>
    where I: MarkovIdentifier + Encodable,
          T: Encodable
{
    fn encode<S: Encoder>(&self, s: &mut S) -> Result<(), S::Error> {
        s.emit_struct("MarkovState", 3, |s| {
            try!(s.emit_struct_field("identifier", 0, |s| self.identifier.encode(s)));
            try!(s.emit_struct_field("transitions", 1, |s| self.transitions.encode(s)));
            try!(s.emit_struct_field("value", 2, |s| self.value.encode(s)));
            Ok(())
        })
    }
}

impl<I, T> Decodable for MarkovState<I, T>
    where I: MarkovIdentifier + Decodable,
          T: Decodable
{
    fn decode<D: Decoder>(d: &mut D) -> Result<MarkovState<I, T>, D::Error> {
        d.read_struct("MarkovState", 3, |d| {
            let identifier = try!(d.read_struct_field("identifier", 0, |d| I::decode(d)));
            let transitions = try!(d.read_struct_field("transitions", 1, |d| HashMap::decode(d)));
            let value = try!(d.read_struct_field("value", 2, |d| T::decode(d)));
            Ok(MarkovState {
                identifier: identifier,
                transitions: transitions,
                value: value,
            })
        })
    }
}

/// An implementation of a Markov Chain.
pub struct MarkovChain<I, T>
    where I: MarkovIdentifier
{
    states: HashMap<I, MarkovState<I, T>>,
}

impl<I, T> MarkovChain<I, T>
    where I: MarkovIdentifier
{
    /// Calculates the transition from a given state, returning the next
    /// state's identifier.
    ///
    /// This function takes as a parameter a mutable reference to a source of
    /// randomness which implements rand::Rng; that source determines which
    /// transition will be chosen.
    pub fn get_next<R: Rng>(&self, id: &I, rng: &mut R) -> Option<I> {
        self.states.get(id).and_then(|state| state.next(&mut *rng))
    }

    /// Returns an iterator over the values of the states of this Markov Chain.
    ///
    /// The returned iterator will perform a walk through the Markov Chain,
    /// starting with the state with the given identifier, and choosing
    /// transitions according to the given source of randomness. The iterator
    /// will return the _values_ of the states it visits, not the states
    /// themselves or their identifiers.
    pub fn get_iter<'a, 'b, R: Rng>(&'a self, id: &I, rng: &'b mut R) -> Iter<'a, 'b, I, T, R> {
        Iter {
            states: &self.states,
            curr_id: Some(id.clone()),
            rng: rng,
        }
    }
}

impl<I, T> Clone for MarkovChain<I, T>
    where I: MarkovIdentifier,
          T: Clone
{
    fn clone(&self) -> Self {
        MarkovChain { states: self.states.clone() }
    }
}

impl<I, T> fmt::Debug for MarkovChain<I, T>
    where I: MarkovIdentifier + fmt::Debug,
          T: fmt::Debug
{
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt.debug_struct("MarkovChain")
            .field("states", &self.states)
            .finish()
    }
}

impl<I, T> PartialEq for MarkovChain<I, T>
    where I: MarkovIdentifier,
          T: PartialEq
{
    fn eq(&self, other: &Self) -> bool {
        self.states == other.states
    }
}

impl<I, T> Eq for MarkovChain<I, T>
    where I: MarkovIdentifier,
          T: Eq
{
}

impl<I, T> Encodable for MarkovChain<I, T>
    where I: MarkovIdentifier + Encodable,
          T: Encodable
{
    fn encode<S: Encoder>(&self, s: &mut S) -> Result<(), S::Error> {
        s.emit_struct("MarkovChain",
                      1,
                      |s| s.emit_struct_field("states", 0, |s| self.states.encode(s)))
    }
}

impl<I, T> Decodable for MarkovChain<I, T>
    where I: MarkovIdentifier + Decodable,
          T: Decodable
{
    fn decode<D: Decoder>(d: &mut D) -> Result<MarkovChain<I, T>, D::Error> {
        d.read_struct("MarkovChain", 1, |d| {
            let states = try!(d.read_struct_field("states", 0, |d| HashMap::decode(d)));
            Ok(MarkovChain { states: states })
        })
    }
}

/// An iterator over the values of a MarkovChain.
///
/// The items returned by this iterator will be the _values_ of each visited
/// state, not their identifiers.
pub struct Iter<'a, 'b, I, T, R>
    where I: 'a + MarkovIdentifier,
          T: 'a,
          R: 'b + Rng
{
    states: &'a HashMap<I, MarkovState<I, T>>,
    curr_id: Option<I>,
    rng: &'b mut R,
}

impl<'a, 'b, I, T, R> Iterator for Iter<'a, 'b, I, T, R>
    where I: 'a + MarkovIdentifier,
          T: 'a,
          R: 'b + Rng
{
    type Item = &'a T;

    fn next(&mut self) -> Option<&'a T> {
        let curr_state = self.curr_id.clone().and_then(|id| self.states.get(&id));
        let ret = curr_state.map(|state| &state.value);
        let next_id = curr_state.and_then(|state| state.next(&mut *self.rng));

        mem::replace(&mut self.curr_id, next_id);

        ret
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::collections::HashMap;
    use rand;
    use rustc_serialize::json::{self, Json};

    // For easy creation of a HashMap
    macro_rules! hashmap {
        ( $($key:expr => $value:expr),+ ) => {
            {
                let mut m = ::std::collections::HashMap::new();
                $(
                    m.insert($key, $value);
                )+
                m
            }
        };
    }

    macro_rules! assert_json_eq {
        ( $left:expr , $right:expr ) => {
            assert_eq!(Json::from_str(&$left).unwrap(),
                       Json::from_str(&$right).unwrap())
        };
        ( $left:expr , $right:expr , $($arg:tt)* ) => {
            assert_eq!(Json::from_str(&$left).unwrap(),
                       Json::from_str(&$right).unwrap(), $($arg)*)
        };
    }

    #[test]
    // Encode a MarkovState into a JSON string
    fn markov_state_encode() {
        let ms = MarkovState {
            identifier: 0,
            transitions: hashmap![
                0 => 10, 1 => 20, 2 => 5
            ],
            value: 'a',
        };
        let ms_json = json::encode(&ms).unwrap();

        let test_json = r#"{
            "identifier": 0,
            "transitions": {
                "0": 10,
                "1": 20,
                "2": 5
            },
            "value": "a"
        }"#;

        assert_json_eq!(ms_json, test_json);
    }

    #[test]
    // Decode a JSON string into a MarkovState
    fn markov_state_decode() {
        let ms = MarkovState {
            identifier: 0,
            transitions: hashmap![
                0 => 10, 1 => 20, 2 => 5
            ],
            value: 'a',
        };
        let test_ms: MarkovState<u32, char> = json::decode(r#"{
            "identifier": 0,
            "transitions": {
                "0": 10,
                "1": 20,
                "2": 5
            },
            "value": "a"
        }"#)
            .unwrap();

        assert_eq!(ms, test_ms);
    }

    #[test]
    // Encode a MarkovChain into a JSON string
    fn markov_chain_encode() {
        let mc = MarkovChain {
            states: hashmap![
                0 => MarkovState {
                    identifier: 0,
                    transitions: hashmap![
                        0 => 10,
                        1 => 20,
                        2 => 5
                    ],
                    value: 'a'
                },
                1 => MarkovState {
                    identifier: 1,
                    transitions: hashmap![
                        0 => 5,
                        1 => 30,
                        2 => 15
                    ],
                    value: 'b'
                },
                2 => MarkovState {
                    identifier: 2,
                    transitions: hashmap![
                        0 => 10,
                        1 => 10
                    ],
                    value: 'c'
                }
            ],
        };
        let mc_json = json::encode(&mc).unwrap();

        let test_json = r#"{
            "states": {
                "0": {
                    "identifier": 0,
                    "transitions": {
                        "0": 10,
                        "1": 20,
                        "2": 5
                    },
                    "value": "a"
                },
                "1": {
                    "identifier": 1,
                    "transitions": {
                        "0": 5,
                        "1": 30,
                        "2": 15
                    },
                    "value": "b"
                },
                "2": {
                    "identifier": 2,
                    "transitions": {
                        "0": 10,
                        "1": 10
                    },
                    "value": "c"
                }
            }
        }"#;

        assert_json_eq!(mc_json, test_json);
    }

    #[test]
    // Decode a JSON string into a MarkovChain
    fn markov_chain_decode() {
        let mc = MarkovChain {
            states: hashmap![
                0 => MarkovState {
                    identifier: 0,
                    transitions: hashmap![
                        0 => 10,
                        1 => 20,
                        2 => 5
                    ],
                    value: 'a'
                },
                1 => MarkovState {
                    identifier: 1,
                    transitions: hashmap![
                        0 => 5,
                        1 => 30,
                        2 => 15
                    ],
                    value: 'b'
                },
                2 => MarkovState {
                    identifier: 2,
                    transitions: hashmap![
                        0 => 10,
                        1 => 10
                    ],
                    value: 'c'
                }
            ],
        };

        let test_mc: MarkovChain<u32, char> = json::decode(r#"{
            "states": {
                "0": {
                    "identifier": 0,
                    "transitions": {
                        "0": 10,
                        "1": 20,
                        "2": 5
                    },
                    "value": "a"
                },
                "1": {
                    "identifier": 1,
                    "transitions": {
                        "0": 5,
                        "1": 30,
                        "2": 15
                    },
                    "value": "b"
                },
                "2": {
                    "identifier": 2,
                    "transitions": {
                        "0": 10,
                        "1": 10
                    },
                    "value": "c"
                }
            }
        }"#)
            .unwrap();

        assert_eq!(mc, test_mc);
    }

    #[test]
    // Iterate over deterministic Markov chain and collect to string
    fn deterministic_markov_chain_to_string() {
        let mc = MarkovChain {
            states: hashmap![
                0 => MarkovState {
                    identifier: 0,
                    transitions: hashmap![
                        1 => 10
                    ],
                    value: 'A'
                },
                1 => MarkovState {
                    identifier: 1,
                    transitions: hashmap![
                        2 => 10
                    ],
                    value: 'r'
                },
                2 => MarkovState {
                    identifier: 2,
                    transitions: hashmap![
                        3 => 10
                    ],
                    value: 'a'
                },
                3 => MarkovState {
                    identifier: 3,
                    transitions: hashmap![
                        4 => 10
                    ],
                    value: 'n'
                },
                4 => MarkovState {
                    identifier: 4,
                    transitions: hashmap![
                        5 => 10
                    ],
                    value: 'd'
                },
                5 => MarkovState {
                    identifier: 5,
                    transitions: hashmap![
                        6 => 10
                    ],
                    value: 'u'
                },
                6 => MarkovState {
                    identifier: 6,
                    transitions: HashMap::new(),
                    value: 'r'
                }
            ],
        };

        let mut rng = rand::thread_rng();

        let result: String = mc.get_iter(&0, &mut rng).map(|&c| c.clone()).collect();

        assert_eq!(result, "Arandur");
    }
}
