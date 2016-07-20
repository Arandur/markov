use std::collections::HashMap;
use std::hash::Hash;
use std::iter::{Iterator, IntoIterator};
use rand::{Rand, Rng};
use rand::distributions::{Weighted, WeightedChoice, IndependentSample};
use rustc_serialize::{Encodable, Decodable, Encoder, Decoder};

pub trait MarkovState: Eq + Hash + Clone + Rand + Encodable + Decodable {}
impl<T> MarkovState for T where T: Eq + Hash + Clone + Rand + Encodable + Decodable {}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MarkovTransitionSet<T> where T: MarkovState {
    transitions: HashMap<T, u32>
}

impl<T> MarkovTransitionSet<T> where T: MarkovState {
    pub fn next<R: Rng>(&self, rng: &mut R) -> Option<T> {
        if self.transitions.is_empty() {
            None
        } else {
            // This is a time-space tradeoff; we could store the WeightedChoice
            // in the struct itself, but that would double the storage needed.
            let mut items = self.transitions.clone().into_iter()
                .map(|(k, v)| Weighted { weight: v, item: k })
                .collect::<Vec<_>>();
            let wc = WeightedChoice::new(&mut items);
            Some(wc.ind_sample(&mut *rng))
        }
    }
}

impl<T> Encodable for MarkovTransitionSet<T> where T: MarkovState {
    fn encode<S: Encoder>(&self, s: &mut S) -> Result<(), S::Error> {
        s.emit_struct("MarkovTransitionSet", 1, |s| {
            s.emit_struct_field(
                "transitions", 0, |s| self.transitions.encode(s))
        })
    }
}

impl<T> Decodable for MarkovTransitionSet<T> where T: MarkovState {
    fn decode<D: Decoder>(d: &mut D) -> Result<MarkovTransitionSet<T>, D::Error> {
        d.read_struct("MarkovTransitionSet", 1, |d| {
            let transitions = try!(
                d.read_struct_field("transitions", 0, |d| HashMap::decode(d)));
            Ok(MarkovTransitionSet { transitions: transitions })
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MarkovChain<T> where T: MarkovState {
    first: T,
    states: HashMap<T, MarkovTransitionSet<T>>
}

impl<T> MarkovChain<T> where T: MarkovState {
    pub fn iter<'a, 'b, R: Rng>(&'b self, rng: &'a mut R) -> Iter<'a, 'b, T, R> {
        Iter {
            current: Some(self.first.clone()),
            rng: rng,
            states: &self.states
        }
    }

    pub fn into_iter<'a, R: Rng>(self, rng: &'a mut R) -> IntoIter<'a, T, R> {
        IntoIter {
            current: Some(self.first),
            rng: rng,
            states: self.states
        }
    }
}

impl<T> Encodable for MarkovChain<T> where T: MarkovState {
    fn encode<S: Encoder>(&self, s: &mut S) -> Result<(), S::Error> {
        s.emit_struct("MarkovChain", 2, |s| {
            try!(s.emit_struct_field("first", 0, |s| self.first.encode(s)));
            try!(s.emit_struct_field("states", 1, |s| self.states.encode(s)));
            Ok(())
        })
    }
}

impl<T> Decodable for MarkovChain<T> where T: MarkovState {
    fn decode<D: Decoder>(d: &mut D) -> Result<MarkovChain<T>, D::Error> {
        d.read_struct("MarkovChain", 2, |d| {
            let first = try!(
                d.read_struct_field("first", 0, |d| T::decode(d)));
            let states = try!(
                d.read_struct_field("states", 1, |d| HashMap::decode(d)));
            Ok(MarkovChain { first: first, states: states })
        })
    }
}

#[derive(Debug)]
pub struct Iter<'a, 'b, T, R> where T: 'b + MarkovState, R: 'a + Rng {
    current: Option<T>,
    rng: &'a mut R,
    states: &'b HashMap<T, MarkovTransitionSet<T>>
}

impl<'a, 'b, T, R> Iterator for Iter<'a, 'b, T, R> 
    where T: 'b + MarkovState, R: 'a + Rng {
    type Item = T;

    fn next(&mut self) -> Option<T> {
        let ret = self.current.clone();
        match &ret {
            &Some(ref state) => {
                self.current = self.states.get(&state)
                    .and_then(|ts| ts.next(&mut *self.rng));
            },
            &None => { self.current = None; }
        };
        ret
    }
}

#[derive(Debug)]
pub struct IntoIter<'a, T, R> where T: MarkovState, R: 'a + Rng {
    current: Option<T>,
    rng: &'a mut R,
    states: HashMap<T, MarkovTransitionSet<T>>
}

impl<'a, 'b, T, R> Iterator for IntoIter<'a, T, R> 
    where T: MarkovState, R: 'a + Rng {
    type Item = T;

    fn next(&mut self) -> Option<T> {
        let ret = self.current.clone();
        match &ret {
            &Some(ref state) => {
                let states_copy = self.states.clone();
                self.current = states_copy.get(&state)
                    .and_then(|ts| ts.next(&mut *self.rng));
            },
            &None => { self.current = None; }
        };
        ret
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use rand;
    use rustc_serialize::json::{self, Json};

    // Convenience function which tests to see if two JSON strings are equal
    // with respect to their structure
    fn json_eq(lhs: &str, rhs: &str) -> bool {
        Json::from_str(lhs) == Json::from_str(rhs)
    }

    // For easy creation of a HashMap
    macro_rules! hashmap(
        { $($key:expr => $value:expr),+ } => {
            {
                let mut m = ::std::collections::HashMap::new();
                $(
                    m.insert($key, $value);
                )+
                m
            }
        };
    );

    #[test]
    // Encode a MarkovTransitionSet into a JSON string
    fn markov_transition_set_encode() {
        let mts = MarkovTransitionSet { 
            transitions: hashmap![ 
                'a' => 10, 'b' => 20, 'c' => 5 
            ]
        };
        let mts_json = json::encode(&mts).unwrap();

        let test_json = r#"{
            "transitions": {
                "a": 10,
                "b": 20,
                "c": 5
            }
        }"#;

        assert!(json_eq(&mts_json, &test_json));
    }

    #[test]
    // Decode a JSON string into a MarkovTransitionSet
    fn markov_transition_set_decode() {
        let mts = MarkovTransitionSet { 
            transitions: hashmap![ 
                'a' => 10, 'b' => 20, 'c' => 5 
            ]
        };
        let test_mts: MarkovTransitionSet<char> = json::decode(r#"{
            "transitions": {
                "a": 10,
                "b": 20,
                "c": 5
            }
        }"#).unwrap();

        assert_eq!(mts, test_mts);
    }

    #[test]
    // Encode a MarkovChain into a JSON string
    fn markov_chain_encode() {
        let mc = MarkovChain {
            first: 'a',
            states: hashmap![
                'a' => MarkovTransitionSet {
                    transitions: hashmap![
                        'a' => 10,
                        'b' => 20,
                        'c' => 5
                    ]
                },
                'b' => MarkovTransitionSet {
                    transitions: hashmap![
                        'a' => 5,
                        'b' => 30,
                        'c' => 15
                    ]
                },
                'c' => MarkovTransitionSet {
                    transitions: hashmap![
                        'a' => 10,
                        'b' => 10
                    ]
                }
            ]
        };
        let mc_json = json::encode(&mc).unwrap();

        let test_json = r#"{
            "first": "a",
            "states": {
                "a": {
                    "transitions": {
                        "a": 10,
                        "b": 20,
                        "c": 5
                    }
                },
                "b": {
                    "transitions": {
                        "a": 5,
                        "b": 30,
                        "c": 15
                    }
                },
                "c": {
                    "transitions": {
                        "a": 10,
                        "b": 10
                    }
                }
            }
        }"#;

        assert!(json_eq(&mc_json, &test_json));
    }

    #[test]
    // Decode a JSON string into a MarkovChain
    fn markov_chain_decode() {
        let mc = MarkovChain {
            first: 'a',
            states: hashmap![
                'a' => MarkovTransitionSet {
                    transitions: hashmap![
                        'a' => 10,
                        'b' => 20,
                        'c' => 5
                    ]
                },
                'b' => MarkovTransitionSet {
                    transitions: hashmap![
                        'a' => 5,
                        'b' => 30,
                        'c' => 15
                    ]
                },
                'c' => MarkovTransitionSet {
                    transitions: hashmap![
                        'a' => 10,
                        'b' => 10
                    ]
                }
            ]
        };

        let test_mc: MarkovChain<char> = json::decode(r#"{
            "first": "a",
            "states": {
                "a": {
                    "transitions": {
                        "a": 10,
                        "b": 20,
                        "c": 5
                    }
                },
                "b": {
                    "transitions": {
                        "a": 5,
                        "b": 30,
                        "c": 15
                    }
                },
                "c": {
                    "transitions": {
                        "a": 10,
                        "b": 10
                    }
                }
            }
        }"#).unwrap();

        assert_eq!(mc, test_mc);
    }

    #[test]
    // Iterate over deterministic Markov chain and collect to string
    fn deterministic_markov_chain_to_string() {
        let mc = MarkovChain {
            first: 'A',
            states: hashmap![
                'A' => MarkovTransitionSet {
                    transitions: hashmap![
                        'r' => 10
                    ]
                },
                'r' => MarkovTransitionSet {
                    transitions: hashmap![
                        'a' => 10
                    ]
                },
                'a' => MarkovTransitionSet {
                    transitions: hashmap![
                        'n' => 10
                    ]
                },
                'n' => MarkovTransitionSet {
                    transitions: hashmap![
                        'd' => 10
                    ]
                },
                'd' => MarkovTransitionSet {
                    transitions: hashmap![
                        'u' => 10
                    ]
                },
                'u' => MarkovTransitionSet {
                    transitions: hashmap![
                        'R' => 10
                    ]
                }
            ]
        };

        let mut rng = rand::thread_rng();

        let result: String = mc.iter(&mut rng).collect();

        assert_eq!(result, "AranduR");
    }
}
