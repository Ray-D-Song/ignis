use std::sync::Arc;

use crate::json_schema::{Dfa, DEAD_STATE};

pub struct VocabIndex {
    pub vocab_size: usize,
    pub bytes: Vec<u8>,
    pub offsets: Vec<u32>,
    pub first_byte_buckets: Vec<Vec<u32>>,
}

impl VocabIndex {
    pub fn build<'a>(vocab_size: usize, get_bytes: impl Fn(u32) -> Option<Vec<u8>>) -> Self {
        let mut bytes = Vec::with_capacity(vocab_size * 5);
        let mut offsets = Vec::with_capacity(vocab_size + 1);
        let mut first_byte_buckets: Vec<Vec<u32>> = vec![Vec::new(); 256];

        offsets.push(0);
        for id in 0..vocab_size {
            if let Some(tok_bytes) = get_bytes(id as u32) {
                if !tok_bytes.is_empty() {
                    let b0 = tok_bytes[0] as usize;
                    first_byte_buckets[b0].push(id as u32);
                    bytes.extend_from_slice(&tok_bytes);
                }
            }
            offsets.push(bytes.len() as u32);
        }

        Self {
            vocab_size,
            bytes,
            offsets,
            first_byte_buckets,
        }
    }

    #[inline(always)]
    pub fn token_bytes(&self, id: u32) -> &[u8] {
        let idx = id as usize;
        if idx >= self.vocab_size {
            return &[];
        }
        let s = self.offsets[idx] as usize;
        let e = self.offsets[idx + 1] as usize;
        &self.bytes[s..e]
    }
}

pub struct TokenFsm {
    dfa: Dfa,
    vocab: Arc<VocabIndex>,
    eos_ids: Vec<u32>,
    state: u32,
    done: bool,
    num_words: usize,
    bitset_cache: Vec<Option<(Vec<u32>, u32)>>,
    empty_bitset: (Vec<u32>, u32),
}

impl TokenFsm {
    pub fn new(dfa: Dfa, vocab: Arc<VocabIndex>, eos_ids: Vec<u32>) -> Self {
        let num_states = dfa.num_states;
        let num_words = (vocab.vocab_size + 31) / 32;
        Self {
            dfa,
            vocab,
            eos_ids,
            state: 0,
            done: false,
            num_words,
            bitset_cache: vec![None; num_states],
            empty_bitset: (vec![0u32; num_words], 0),
        }
    }

    pub fn num_dfa_states(&self) -> usize {
        self.dfa.num_states
    }

    fn compute_or_get_bitset(&mut self, state: u32) -> (&[u32], u32) {
        if self.done || state == DEAD_STATE || (state as usize) >= self.dfa.num_states {
            return (&self.empty_bitset.0, 0);
        }
        let idx = state as usize;
        if self.bitset_cache[idx].is_none() {
            let mut words = vec![0u32; self.num_words];
            let mut count = 0u32;

            for b in 0u16..256 {
                let b_u8 = b as u8;
                let s1 = self.dfa.step_byte(state, b_u8);
                if s1 == DEAD_STATE {
                    continue;
                }
                let bucket = &self.vocab.first_byte_buckets[b as usize];
                for &tok_id in bucket {
                    let s = self.vocab.offsets[tok_id as usize] as usize;
                    let e = self.vocab.offsets[tok_id as usize + 1] as usize;
                    let tail = &self.vocab.bytes[s + 1..e];
                    if self.dfa.step_bytes(s1, tail) != DEAD_STATE {
                        let w = (tok_id >> 5) as usize;
                        let bit = 1u32 << (tok_id & 31);
                        if (words[w] & bit) == 0 {
                            words[w] |= bit;
                            count += 1;
                        }
                    }
                }
            }

            if self.dfa.is_accepting(state) {
                for &eos in &self.eos_ids {
                    if (eos as usize) < self.vocab.vocab_size {
                        let w = (eos >> 5) as usize;
                        let bit = 1u32 << (eos & 31);
                        if (words[w] & bit) == 0 {
                            words[w] |= bit;
                            count += 1;
                        }
                    }
                }
            }

            self.bitset_cache[idx] = Some((words, count));
        }
        let entry = self.bitset_cache[idx].as_ref().unwrap();
        (&entry.0, entry.1)
    }

    pub fn allowed_bitset(&mut self) -> (&[u32], u32) {
        self.compute_or_get_bitset(self.state)
    }

    pub fn mask_logits(&mut self, logits: &mut [f32]) -> u32 {
        let len = logits.len().min(self.vocab.vocab_size);
        let (words, count) = self.compute_or_get_bitset(self.state);
        for (wi, &mask) in words.iter().enumerate() {
            let base = wi << 5;
            if base >= len {
                break;
            }
            if mask == u32::MAX {
                continue;
            }
            let end = (base + 32).min(len);
            if mask == 0 {
                for slot in &mut logits[base..end] {
                    *slot = f32::NEG_INFINITY;
                }
                continue;
            }
            for bit in 0..(end - base) {
                if (mask & (1u32 << bit)) == 0 {
                    logits[base + bit] = f32::NEG_INFINITY;
                }
            }
        }
        if logits.len() > self.vocab.vocab_size {
            for slot in &mut logits[self.vocab.vocab_size..] {
                *slot = f32::NEG_INFINITY;
            }
        }
        count
    }

    pub fn advance(&mut self, token_id: u32) -> bool {
        if self.done || self.state == DEAD_STATE {
            return false;
        }
        if self.eos_ids.contains(&token_id) && self.dfa.is_accepting(self.state) {
            self.done = true;
            return true;
        }
        let bytes = self.vocab.token_bytes(token_id);
        if bytes.is_empty() {
            return false;
        }
        let next = self.dfa.step_bytes(self.state, bytes);
        if next == DEAD_STATE {
            return false;
        }
        self.state = next;
        true
    }

    pub fn is_done(&self) -> bool {
        self.done || (self.state != DEAD_STATE && self.dfa.is_accepting(self.state))
    }

    pub fn reset(&mut self) {
        self.state = 0;
        self.done = false;
    }
}
