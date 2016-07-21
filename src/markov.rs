use std::collections::HashMap;
use std::fmt;
use std::hash::Hash;
use std::iter::Iterator;
use std::mem;
use rand::Rng;
use rand::distributions::{Weighted, WeightedChoice, IndependentSample};
use rustc_serialize::{Encodable, Decodable, Encoder, Decoder};

pub trait MarkovIdentifier: Clone + Decodable + Encodable + Eq + Hash {}
impl<T> MarkovIdentifier for T
    where T: Clone + Decodable + Encodable + Eq + Hash {}

pub trait MarkovValue: Decodable + Encodable {}
impl<T> MarkovValue for T
    where T: Decodable + Encodable {}

pub struct MarkovState<I: MarkovIdentifier, T: MarkovValue> {
    identifier: I,
    transitions: HashMap<I, u32>,
    pub value: T,
}

impl<I: MarkovIdentifier, T: MarkovValue> MarkovState<I, T> {
    fn next<R: Rng>(&self, rng: &mut R) -> Option<I> {
        if self.transitions.is_empty() {
            None
        } else {
            let mut items: Vec<_> = self.transitions.clone().into_iter()
                .map(|(k, v)| Weighted { weight: v, item: k })
                .collect();
            let wc = WeightedChoice::new(&mut items);
            Some(wc.ind_sample(&mut *rng))
        }
    }
}

impl<I: MarkovIdentifier, T: MarkovValue + Clone> Clone for MarkovState<I, T> {
    fn clone(&self) -> Self {
        MarkovState {
            identifier: self.identifier.clone(),
            transitions: self.transitions.clone(),
            value: self.value.clone()
        }
    }
}

impl<I: MarkovIdentifier + fmt::Debug, T: MarkovValue + fmt::Debug> fmt::Debug for MarkovState<I, T> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt.debug_struct("MarkovState")
            .field("identifier", &self.identifier)
            .field("transitions", &self.transitions)
            .field("value", &self.value)
            .finish()
    }
}

impl<I: MarkovIdentifier, T: MarkovValue + PartialEq> PartialEq for MarkovState<I, T> {
    fn eq(&self, other: &Self) -> bool {
        self.identifier == other.identifier &&
            self.transitions == other.transitions &&
            self.value == other.value
    }
}

impl<I: MarkovIdentifier, T: MarkovValue + Eq> Eq for MarkovState<I, T> {}

impl<I: MarkovIdentifier, T: MarkovValue> Encodable for MarkovState<I, T> {
    fn encode<S: Encoder>(&self, s: &mut S) -> Result<(), S::Error> {
        s.emit_struct("MarkovState", 3, |s| {
            try!(s.emit_struct_field("identifier", 0, 
                                     |s| self.identifier.encode(s)));
            try!(s.emit_struct_field("transitions", 1,
                                     |s| self.transitions.encode(s)));
            try!(s.emit_struct_field("value", 2,
                                     |s| self.value.encode(s)));
            Ok(())
        })
    }
}

impl<I: MarkovIdentifier, T: MarkovValue> Decodable for MarkovState<I, T> {
    fn decode<D: Decoder>(d: &mut D) -> Result<MarkovState<I, T>, D::Error> {
        d.read_struct("MarkovState", 3, |d| {
            let identifier = try!(
                d.read_struct_field("identifier", 0, |d| I::decode(d)));
            let transitions = try!(
                d.read_struct_field("transitions", 1, |d| HashMap::decode(d)));
            let value = try!(
                d.read_struct_field("value", 2, |d| T::decode(d)));
            Ok(MarkovState {
                identifier: identifier,
                transitions: transitions,
                value: value
            })
        })
    }
}

pub struct MarkovChain<I: MarkovIdentifier, T: MarkovValue> {
    states: HashMap<I, MarkovState<I, T>>
}

impl<I: MarkovIdentifier, T: MarkovValue> MarkovChain<I, T> {
    pub fn get_state(&self, id: &I) -> Option<&MarkovState<I, T>> {
        self.states.get(id)
    }

    pub fn get_next<R: Rng>(&self, id: &I, rng: &mut R) -> Option<I> {
        self.states.get(id).and_then(|state| state.next(&mut *rng))
    }

    pub fn get_iter<'a, 'b, R: Rng>(&'a self, id: &I, rng: &'b mut R) -> Iter<'a, 'b, I, T, R> {
        Iter { states: &self.states, curr_id: Some(id.clone()), rng: rng }
    }
}

impl<I: MarkovIdentifier, T: MarkovValue + Clone> Clone for MarkovChain<I, T> {
    fn clone(&self) -> Self {
        MarkovChain {
            states: self.states.clone()
        }
    }
}

impl<I: MarkovIdentifier + fmt::Debug, T: MarkovValue + fmt::Debug> fmt::Debug for MarkovChain<I, T> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt.debug_struct("MarkovChain")
            .field("states", &self.states)
            .finish()
    }
}

impl<I: MarkovIdentifier, T: MarkovValue + PartialEq> PartialEq for MarkovChain<I, T> {
    fn eq(&self, other: &Self) -> bool {
        self.states == other.states
    }
}

impl<I: MarkovIdentifier, T: MarkovValue + Eq> Eq for MarkovChain<I, T> {}

impl<I: MarkovIdentifier, T: MarkovValue> Encodable for MarkovChain<I, T> {
    fn encode<S: Encoder>(&self, s: &mut S) -> Result<(), S::Error> {
        s.emit_struct("MarkovChain", 1, 
                      |s| s.emit_struct_field("states", 0, 
                                              |s| self.states.encode(s)))
    }
}

impl<I: MarkovIdentifier, T: MarkovValue> Decodable for MarkovChain<I, T> {
    fn decode<D: Decoder>(d: &mut D) -> Result<MarkovChain<I, T>, D::Error> {
        d.read_struct("MarkovChain", 1, |d| {
            let states = try!(
                d.read_struct_field("states", 0, |d| HashMap::decode(d)));
            Ok(MarkovChain {
                states: states
            })
        })
    }
}

pub struct Iter<'a, 'b, I: 'a + MarkovIdentifier, T: 'a + MarkovValue, R: 'b + Rng> {
    states: &'a HashMap<I, MarkovState<I, T>>,
    curr_id: Option<I>,
    rng: &'b mut R
}

impl<'a, 'b, I: 'a + MarkovIdentifier, T: 'a + MarkovValue, R: 'b + Rng> Iterator for Iter<'a, 'b, I, T, R> {
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
            value: 'a'
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
            value: 'a'
        };
        let test_ms: MarkovState<u32, char> = json::decode(r#"{
            "identifier": 0,
            "transitions": {
                "0": 10,
                "1": 20,
                "2": 5
            },
            "value": "a"
        }"#).unwrap();

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
            ]
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
            ]
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
        }"#).unwrap();

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
            ]
        };

        let mut rng = rand::thread_rng();

        let result: String = mc.get_iter(&0, &mut rng).map(|&c| c.clone()).collect();

        assert_eq!(result, "Arandur");
    }
}
