use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::slice;
use std::sync::{Arc, OnceLock};

use fancy_regex::Regex;
use serde::Deserialize;

pub mod fsm;
pub mod json_schema;

use fsm::{TokenFsm, VocabIndex};
use json_schema::Dfa;

#[derive(Deserialize)]
struct TokenizerJson {
    #[serde(default)]
    added_tokens: Vec<AddedTokenJson>,
    normalizer: Option<NormalizerJson>,
    pre_tokenizer: Option<PreTokenizerJson>,
    decoder: Option<DecoderJson>,
    model: ModelJson,
}

#[derive(Deserialize)]
#[serde(tag = "type")]
enum NormalizerJson {
    Sequence {
        normalizers: Vec<NormalizerJson>,
    },
    Replace {
        pattern: PatternJson,
        content: String,
    },
    #[serde(other)]
    Unknown,
}

#[derive(Deserialize)]
#[serde(tag = "type")]
enum DecoderJson {
    Sequence {
        decoders: Vec<DecoderJson>,
    },
    Replace {
        pattern: PatternJson,
        content: String,
    },
    ByteFallback {},
    ByteLevel {},
    #[serde(other)]
    Unknown,
}

#[derive(Deserialize)]
struct AddedTokenJson {
    id: u32,
    content: String,
    #[serde(default)]
    lstrip: bool,
    #[serde(default)]
    rstrip: bool,
    #[serde(default)]
    single_word: bool,
    #[serde(default)]
    special: bool,
}

#[derive(Deserialize)]
#[serde(tag = "type")]
enum PreTokenizerJson {
    Sequence {
        pretokenizers: Vec<PreTokenizerJson>,
    },
    Split {
        pattern: PatternJson,
        #[allow(dead_code)]
        #[serde(default)]
        behavior: String,
        #[serde(default)]
        invert: bool,
    },
    Digits {
        #[serde(default)]
        individual_digits: bool,
    },
    ByteLevel {
        #[serde(default)]
        add_prefix_space: bool,
        #[serde(default)]
        use_regex: bool,
    },
    #[serde(other)]
    Unknown,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum PatternJson {
    Regex { #[serde(rename = "Regex")] regex: String },
    String { #[serde(rename = "String")] literal: String },
}

#[derive(Deserialize)]
#[serde(untagged)]
enum MergeEntry {
    Str(String),
    Pair(String, String),
}

#[derive(Deserialize)]
struct ModelJson {
    vocab: HashMap<String, u32>,
    merges: Vec<MergeEntry>,
    #[serde(default)]
    byte_fallback: bool,
    #[serde(default)]
    unk_token: Option<String>,
}

enum NormStep {
    ReplaceLiteral { from: String, to: String },
    ReplaceRegex { regex: Regex, to: String },
}

enum PreStep {
    SplitRegex { regex: Regex, #[allow(dead_code)] invert: bool },
    SplitLiteral { literal: String, behavior: String, #[allow(dead_code)] invert: bool },
    IndividualDigits,
    ByteLevel { add_prefix_space: bool, use_regex: bool },
}

struct AddedTokenEntry {
    id: u32,
    content: String,
    lstrip: bool,
    rstrip: bool,
    single_word: bool,
}

pub struct Tokenizer {
    id_to_token: Vec<String>,
    #[allow(dead_code)]
    byte_to_char: [char; 256],
    byte_tok_ids: [u32; 256],
    char_to_byte: HashMap<char, u8>,
    single_char_ids: HashMap<char, u32>,
    sp_byte_fallback_ids: [u32; 256],
    sp_byte_rev: HashMap<u32, u8>,
    unk_id: u32,
    byte_fallback: bool,
    use_byte_level: bool,
    merges: HashMap<(u32, u32), (u32, u32)>, // (a, b) -> (rank, merged_id)
    added_tokens: Vec<AddedTokenEntry>,
    added_set: HashMap<u32, (String, bool)>,
    norm_steps: Vec<NormStep>,
    pre_steps: Vec<PreStep>,
    dec_replaces: Vec<(String, String)>,
    gpt2_default_re: Option<Regex>,
    vocab_index: OnceLock<Arc<VocabIndex>>,
}

fn build_byte_tables() -> ([char; 256], HashMap<char, u8>) {
    let mut bs: Vec<u32> = Vec::with_capacity(256);
    bs.extend(33u32..=126);
    bs.extend(161u32..=172);
    bs.extend(174u32..=255);
    let mut cs = bs.clone();
    let mut n = 0u32;
    for b in 0u32..256 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    let mut b2c = ['\0'; 256];
    let mut c2b = HashMap::with_capacity(256);
    for (b, c) in bs.into_iter().zip(cs.into_iter()) {
        let ch = char::from_u32(c).unwrap();
        b2c[b as usize] = ch;
        c2b.insert(ch, b as u8);
    }
    (b2c, c2b)
}

fn compile_norm_steps(n: &NormalizerJson, out: &mut Vec<NormStep>) {
    match n {
        NormalizerJson::Sequence { normalizers } => {
            for item in normalizers {
                compile_norm_steps(item, out);
            }
        }
        NormalizerJson::Replace { pattern, content } => match pattern {
            PatternJson::String { literal } => {
                out.push(NormStep::ReplaceLiteral { from: literal.clone(), to: content.clone() });
            }
            PatternJson::Regex { regex } => {
                if let Ok(re) = Regex::new(regex) {
                    out.push(NormStep::ReplaceRegex { regex: re, to: content.clone() });
                }
            }
        },
        NormalizerJson::Unknown => {}
    }
}

fn inspect_decoder(d: &DecoderJson, replaces: &mut Vec<(String, String)>, has_byte_level: &mut bool) {
    match d {
        DecoderJson::Sequence { decoders } => {
            for item in decoders {
                inspect_decoder(item, replaces, has_byte_level);
            }
        }
        DecoderJson::Replace { pattern, content } => {
            if let PatternJson::String { literal } = pattern {
                replaces.push((literal.clone(), content.clone()));
            }
        }
        DecoderJson::ByteLevel {} => *has_byte_level = true,
        DecoderJson::ByteFallback {} | DecoderJson::Unknown => {}
    }
}

fn compile_pre_steps(pt: &PreTokenizerJson, out: &mut Vec<PreStep>, has_byte_level: &mut bool) -> Result<(), String> {
    match pt {
        PreTokenizerJson::Sequence { pretokenizers } => {
            for p in pretokenizers {
                compile_pre_steps(p, out, has_byte_level)?;
            }
        }
        PreTokenizerJson::Split { pattern, behavior, invert } => {
            match pattern {
                PatternJson::Regex { regex } => {
                    let re = Regex::new(regex).map_err(|e| format!("Regex compile error ({regex}): {e}"))?;
                    out.push(PreStep::SplitRegex { regex: re, invert: *invert });
                }
                PatternJson::String { literal } => {
                    out.push(PreStep::SplitLiteral {
                        literal: literal.clone(),
                        behavior: behavior.clone(),
                        invert: *invert,
                    });
                }
            }
        }
        PreTokenizerJson::Digits { individual_digits } => {
            if *individual_digits {
                out.push(PreStep::IndividualDigits);
            }
        }
        PreTokenizerJson::ByteLevel { add_prefix_space, use_regex } => {
            *has_byte_level = true;
            out.push(PreStep::ByteLevel {
                add_prefix_space: *add_prefix_space,
                use_regex: *use_regex,
            });
        }
        PreTokenizerJson::Unknown => {}
    }
    Ok(())
}

impl Tokenizer {
    fn from_json(json_str: &str) -> Result<Self, String> {
        let parsed: TokenizerJson = serde_json::from_str(json_str).map_err(|e| format!("JSON parse error: {e}"))?;
        let (byte_to_char, char_to_byte) = build_byte_tables();

        let mut max_id = parsed.model.vocab.values().copied().max().unwrap_or(0);
        for at in &parsed.added_tokens {
            if at.id > max_id {
                max_id = at.id;
            }
        }
        let mut id_to_token = vec![String::new(); (max_id + 1) as usize];
        for (tok, &id) in &parsed.model.vocab {
            id_to_token[id as usize] = tok.clone();
        }
        let mut added_set = HashMap::new();
        let mut added_tokens: Vec<AddedTokenEntry> = parsed
            .added_tokens
            .into_iter()
            .map(|a| {
                if (a.id as usize) < id_to_token.len() {
                    id_to_token[a.id as usize] = a.content.clone();
                }
                added_set.insert(a.id, (a.content.clone(), a.special));
                AddedTokenEntry {
                    id: a.id,
                    content: a.content,
                    lstrip: a.lstrip,
                    rstrip: a.rstrip,
                    single_word: a.single_word,
                }
            })
            .collect();
        // Longest match first
        added_tokens.sort_by(|a, b| b.content.len().cmp(&a.content.len()));

        let unk_id = parsed
            .model
            .unk_token
            .as_ref()
            .and_then(|u| parsed.model.vocab.get(u).copied())
            .unwrap_or(0);

        let mut single_char_ids = HashMap::new();
        for (tok, &id) in &parsed.model.vocab {
            let mut it = tok.chars();
            if let (Some(ch), None) = (it.next(), it.next()) {
                single_char_ids.insert(ch, id);
            }
        }

        let mut byte_tok_ids = [0u32; 256];
        let mut sp_byte_fallback_ids = [unk_id; 256];
        let mut sp_byte_rev = HashMap::new();
        for b in 0..256 {
            let mut s = String::new();
            s.push(byte_to_char[b]);
            if let Some(&id) = parsed.model.vocab.get(&s) {
                byte_tok_ids[b] = id;
            }
            let hex_tok = format!("<0x{:02X}>", b);
            if let Some(&id) = parsed.model.vocab.get(&hex_tok) {
                sp_byte_fallback_ids[b] = id;
                sp_byte_rev.insert(id, b as u8);
            }
        }

        let mut merges = HashMap::with_capacity(parsed.model.merges.len());
        for (rank, m) in parsed.model.merges.iter().enumerate() {
            let (left, right) = match m {
                MergeEntry::Str(s) => {
                    let mut it = s.splitn(2, ' ');
                    let a = it.next().unwrap_or("");
                    let b = it.next().unwrap_or("");
                    (a, b)
                }
                MergeEntry::Pair(a, b) => (a.as_str(), b.as_str()),
            };
            if let (Some(&lid), Some(&rid)) = (parsed.model.vocab.get(left), parsed.model.vocab.get(right)) {
                let mut merged = String::with_capacity(left.len() + right.len());
                merged.push_str(left);
                merged.push_str(right);
                if let Some(&mid) = parsed.model.vocab.get(&merged) {
                    merges.insert((lid, rid), (rank as u32, mid));
                }
            }
        }

        let mut norm_steps = Vec::new();
        if let Some(norm) = &parsed.normalizer {
            compile_norm_steps(norm, &mut norm_steps);
        }

        let mut pre_steps = Vec::new();
        let mut pre_has_bl = false;
        if let Some(pt) = &parsed.pre_tokenizer {
            compile_pre_steps(pt, &mut pre_steps, &mut pre_has_bl)?;
        }

        let mut dec_replaces = Vec::new();
        let mut dec_has_bl = false;
        if let Some(dec) = &parsed.decoder {
            inspect_decoder(dec, &mut dec_replaces, &mut dec_has_bl);
        }

        let use_byte_level = pre_has_bl || (!parsed.model.byte_fallback && dec_has_bl) || (!parsed.model.byte_fallback && dec_replaces.is_empty());

        let gpt2_default_re = Regex::new(
            r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
        ).ok();

        Ok(Self {
            id_to_token,
            byte_to_char,
            byte_tok_ids,
            char_to_byte,
            single_char_ids,
            sp_byte_fallback_ids,
            sp_byte_rev,
            unk_id,
            byte_fallback: parsed.model.byte_fallback,
            use_byte_level,
            merges,
            added_tokens,
            added_set,
            norm_steps,
            pre_steps,
            dec_replaces,
            gpt2_default_re,
            vocab_index: OnceLock::new(),
        })
    }

    fn apply_regex_isolated(re: &Regex, piece: &str, out: &mut Vec<String>) {
        let mut last = 0usize;
        let bytes = piece.as_bytes();
        let mut pos = 0usize;
        while pos <= bytes.len() {
            match re.find_from_pos(piece, pos) {
                Ok(Some(m)) => {
                    if m.start() > last {
                        out.push(piece[last..m.start()].to_string());
                    }
                    if m.end() > m.start() {
                        out.push(piece[m.start()..m.end()].to_string());
                        last = m.end();
                        pos = m.end();
                    } else {
                        if pos >= bytes.len() {
                            break;
                        }
                        let ch_len = piece[pos..].chars().next().map(|c| c.len_utf8()).unwrap_or(1);
                        pos += ch_len;
                    }
                }
                _ => break,
            }
        }
        if last < piece.len() {
            out.push(piece[last..].to_string());
        }
    }

    fn normalize(&self, text: &str) -> String {
        if self.norm_steps.is_empty() {
            return text.to_string();
        }
        let mut cur = text.to_string();
        for step in &self.norm_steps {
            match step {
                NormStep::ReplaceLiteral { from, to } => {
                    if !from.is_empty() && cur.contains(from.as_str()) {
                        cur = cur.replace(from.as_str(), to.as_str());
                    }
                }
                NormStep::ReplaceRegex { regex, to } => {
                    cur = regex.replace_all(&cur, to.as_str()).into_owned();
                }
            }
        }
        cur
    }

    fn pretokenize(&self, text: &str) -> Vec<String> {
        let mut pieces = vec![self.normalize(text)];
        for step in &self.pre_steps {
            let mut next = Vec::with_capacity(pieces.len() * 2);
            match step {
                PreStep::SplitRegex { regex, invert: _ } => {
                    for p in &pieces {
                        Self::apply_regex_isolated(regex, p, &mut next);
                    }
                }
                PreStep::SplitLiteral { literal, behavior, invert: _ } => {
                    for p in &pieces {
                        if literal.is_empty() {
                            next.push(p.clone());
                            continue;
                        }
                        if !p.contains(literal.as_str()) {
                            next.push(p.clone());
                            continue;
                        }
                        let mut last = 0;
                        let merged_prev = behavior == "MergedWithPrevious";
                        for (idx, _) in p.match_indices(literal.as_str()) {
                            if merged_prev {
                                next.push(p[last..idx + literal.len()].to_string());
                            } else {
                                if idx > last {
                                    next.push(p[last..idx].to_string());
                                }
                                next.push(literal.clone());
                            }
                            last = idx + literal.len();
                        }
                        if last < p.len() {
                            next.push(p[last..].to_string());
                        }
                    }
                }
                PreStep::IndividualDigits => {
                    for p in &pieces {
                        let mut start = 0usize;
                        for (idx, ch) in p.char_indices() {
                            if ch.is_ascii_digit() || ch.is_numeric() {
                                if idx > start {
                                    next.push(p[start..idx].to_string());
                                }
                                let end = idx + ch.len_utf8();
                                next.push(p[idx..end].to_string());
                                start = end;
                            }
                        }
                        if start < p.len() {
                            next.push(p[start..].to_string());
                        }
                    }
                }
                PreStep::ByteLevel { add_prefix_space, use_regex } => {
                    for (idx, p) in pieces.iter().enumerate() {
                        let owned;
                        let s = if *add_prefix_space && idx == 0 && !p.starts_with(' ') {
                            owned = format!(" {p}");
                            owned.as_str()
                        } else {
                            p.as_str()
                        };
                        if *use_regex {
                            if let Some(re) = &self.gpt2_default_re {
                                Self::apply_regex_isolated(re, s, &mut next);
                                continue;
                            }
                        }
                        next.push(s.to_string());
                    }
                }
            }
            pieces = next;
        }
        pieces
    }

    fn bpe_encode_piece(&self, piece: &str, out: &mut Vec<u32>) {
        if piece.is_empty() {
            return;
        }
        let mut ids: Vec<u32> = if self.use_byte_level {
            let bytes = piece.as_bytes();
            if bytes.len() == 1 {
                out.push(self.byte_tok_ids[bytes[0] as usize]);
                return;
            }
            bytes.iter().map(|&b| self.byte_tok_ids[b as usize]).collect()
        } else {
            let mut v = Vec::with_capacity(piece.chars().count());
            for ch in piece.chars() {
                if let Some(&id) = self.single_char_ids.get(&ch) {
                    v.push(id);
                } else if self.byte_fallback {
                    let mut buf = [0u8; 4];
                    for &b in ch.encode_utf8(&mut buf).as_bytes() {
                        v.push(self.sp_byte_fallback_ids[b as usize]);
                    }
                } else {
                    v.push(self.unk_id);
                }
            }
            v
        };

        if ids.len() < 2 {
            out.extend_from_slice(&ids);
            return;
        }
        if ids.len() <= 64 {
            loop {
                if ids.len() < 2 {
                    break;
                }
                let mut best_rank = u32::MAX;
                let mut best_idx = 0usize;
                let mut best_merged = 0u32;
                for i in 0..(ids.len() - 1) {
                    if let Some(&(rank, merged_id)) = self.merges.get(&(ids[i], ids[i + 1])) {
                        if rank < best_rank {
                            best_rank = rank;
                            best_idx = i;
                            best_merged = merged_id;
                        }
                    }
                }
                if best_rank == u32::MAX {
                    break;
                }
                ids[best_idx] = best_merged;
                ids.remove(best_idx + 1);
            }
            out.extend_from_slice(&ids);
            return;
        }

        // O(N log N) linked-list + min-heap BPE for unsplit pieces
        let n = ids.len();
        let mut prev: Vec<isize> = (0..n).map(|i| i as isize - 1).collect();
        let mut next: Vec<isize> = (0..n).map(|i| if i + 1 < n { (i + 1) as isize } else { -1 }).collect();
        let mut alive = vec![true; n];
        let mut gen = vec![0u32; n];
        let mut heap: BinaryHeap<Reverse<(u32, usize, u32, u32)>> = BinaryHeap::with_capacity(n);

        for i in 0..(n - 1) {
            if let Some(&(rank, _)) = self.merges.get(&(ids[i], ids[i + 1])) {
                heap.push(Reverse((rank, i, 0, 0)));
            }
        }
        while let Some(Reverse((_, li, lg, rg))) = heap.pop() {
            if !alive[li] || gen[li] != lg {
                continue;
            }
            let ri = next[li];
            if ri < 0 {
                continue;
            }
            let ri = ri as usize;
            if !alive[ri] || gen[ri] != rg {
                continue;
            }
            let Some(&(_, merged_id)) = self.merges.get(&(ids[li], ids[ri])) else {
                continue;
            };
            ids[li] = merged_id;
            gen[li] += 1;
            alive[ri] = false;
            let nri = next[ri];
            next[li] = nri;
            if nri >= 0 {
                prev[nri as usize] = li as isize;
            }
            if prev[li] >= 0 {
                let pi = prev[li] as usize;
                if let Some(&(r, _)) = self.merges.get(&(ids[pi], ids[li])) {
                    heap.push(Reverse((r, pi, gen[pi], gen[li])));
                }
            }
            if next[li] >= 0 {
                let ni = next[li] as usize;
                if let Some(&(r, _)) = self.merges.get(&(ids[li], ids[ni])) {
                    heap.push(Reverse((r, li, gen[li], gen[ni])));
                }
            }
        }
        let mut cur = 0isize;
        while cur >= 0 {
            let idx = cur as usize;
            if alive[idx] {
                out.push(ids[idx]);
            }
            cur = next[idx];
        }
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        let mut pos = 0usize;
        let mut plain_start = 0usize;

        while pos < text.len() {
            let rest = &text[pos..];
            let mut matched: Option<&AddedTokenEntry> = None;
            for at in &self.added_tokens {
                if rest.starts_with(&at.content) {
                    if at.single_word {
                        let prev_ok = pos == 0 || !text[..pos].chars().last().map(|c| c.is_alphanumeric() || c == '_').unwrap_or(false);
                        let next_ok = pos + at.content.len() == text.len()
                            || !text[pos + at.content.len()..].chars().next().map(|c| c.is_alphanumeric() || c == '_').unwrap_or(false);
                        if !(prev_ok && next_ok) {
                            continue;
                        }
                    }
                    matched = Some(at);
                    break;
                }
            }
            if let Some(at) = matched {
                let mut seg_end = pos;
                if at.lstrip {
                    while seg_end > plain_start {
                        if let Some(ch) = text[plain_start..seg_end].chars().last() {
                            if ch.is_whitespace() {
                                seg_end -= ch.len_utf8();
                                continue;
                            }
                        }
                        break;
                    }
                }
                if seg_end > plain_start {
                    for piece in self.pretokenize(&text[plain_start..seg_end]) {
                        self.bpe_encode_piece(&piece, &mut out);
                    }
                }
                out.push(at.id);
                pos += at.content.len();
                if at.rstrip {
                    while pos < text.len() {
                        if let Some(ch) = text[pos..].chars().next() {
                            if ch.is_whitespace() {
                                pos += ch.len_utf8();
                                continue;
                            }
                        }
                        break;
                    }
                }
                plain_start = pos;
            } else {
                let ch_len = rest.chars().next().map(|c| c.len_utf8()).unwrap_or(1);
                pos += ch_len;
            }
        }
        if plain_start < text.len() {
            for piece in self.pretokenize(&text[plain_start..]) {
                self.bpe_encode_piece(&piece, &mut out);
            }
        }
        out
    }

    pub fn decode(&self, ids: &[u32], skip_special: bool) -> String {
        let mut raw_bytes = Vec::with_capacity(ids.len() * 4);
        for &id in ids {
            if let Some((s, is_special)) = self.added_set.get(&id) {
                if !skip_special || !*is_special {
                    raw_bytes.extend_from_slice(s.as_bytes());
                }
                continue;
            }
            if !self.use_byte_level {
                if self.byte_fallback {
                    if let Some(&b) = self.sp_byte_rev.get(&id) {
                        raw_bytes.push(b);
                        continue;
                    }
                }
                if let Some(tok) = self.id_to_token.get(id as usize) {
                    if self.dec_replaces.is_empty() {
                        raw_bytes.extend_from_slice(tok.as_bytes());
                    } else {
                        let mut s = tok.clone();
                        for (from, to) in &self.dec_replaces {
                            if !from.is_empty() && s.contains(from.as_str()) {
                                s = s.replace(from.as_str(), to.as_str());
                            }
                        }
                        raw_bytes.extend_from_slice(s.as_bytes());
                    }
                }
                continue;
            }
            if let Some(tok) = self.id_to_token.get(id as usize) {
                for ch in tok.chars() {
                    if let Some(&b) = self.char_to_byte.get(&ch) {
                        raw_bytes.push(b);
                    } else {
                        let mut buf = [0u8; 4];
                        raw_bytes.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                    }
                }
            }
        }
        String::from_utf8_lossy(&raw_bytes).into_owned()
    }

    pub fn vocab_size(&self) -> usize {
        self.id_to_token.len()
    }

    pub fn token_raw_bytes(&self, id: u32) -> Option<Vec<u8>> {
        if let Some((s, is_special)) = self.added_set.get(&id) {
            if *is_special || s.is_empty() {
                return None;
            }
            return Some(s.as_bytes().to_vec());
        }
        if !self.use_byte_level {
            if self.byte_fallback {
                if let Some(&b) = self.sp_byte_rev.get(&id) {
                    return Some(vec![b]);
                }
            }
            let tok = self.id_to_token.get(id as usize)?;
            if tok.is_empty() {
                return None;
            }
            if self.dec_replaces.is_empty() {
                return Some(tok.as_bytes().to_vec());
            }
            let mut s = tok.clone();
            for (from, to) in &self.dec_replaces {
                if !from.is_empty() && s.contains(from.as_str()) {
                    s = s.replace(from.as_str(), to.as_str());
                }
            }
            if s.is_empty() {
                return None;
            }
            return Some(s.into_bytes());
        }
        let tok = self.id_to_token.get(id as usize)?;
        if tok.is_empty() {
            return None;
        }
        let mut raw_bytes = Vec::with_capacity(tok.len());
        for ch in tok.chars() {
            if let Some(&b) = self.char_to_byte.get(&ch) {
                raw_bytes.push(b);
            } else {
                let mut buf = [0u8; 4];
                raw_bytes.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
            }
        }
        if raw_bytes.is_empty() { None } else { Some(raw_bytes) }
    }

    pub fn vocab_index(&self) -> Arc<VocabIndex> {
        self.vocab_index
            .get_or_init(|| Arc::new(VocabIndex::build(self.vocab_size(), |id| self.token_raw_bytes(id))))
            .clone()
    }

    pub fn create_fsm(&self, schema_json: &str, eos_ids: &[u32]) -> Result<TokenFsm, String> {
        let dfa = Dfa::from_json_schema_str(schema_json)?;
        let vocab = self.vocab_index();
        Ok(TokenFsm::new(dfa, vocab, eos_ids.to_vec()))
    }
}

#[no_mangle]
pub extern "C" fn ignis_alloc(len: usize) -> *mut u8 {
    let mut buf = Vec::<u8>::with_capacity(len);
    let ptr = buf.as_mut_ptr();
    std::mem::forget(buf);
    ptr
}

#[no_mangle]
pub unsafe extern "C" fn ignis_dealloc(ptr: *mut u8, cap: usize) {
    if !ptr.is_null() && cap > 0 {
        drop(Vec::from_raw_parts(ptr, 0, cap));
    }
}

#[no_mangle]
pub unsafe extern "C" fn ignis_tok_create(json_ptr: *const u8, json_len: usize) -> *mut Tokenizer {
    let bytes = slice::from_raw_parts(json_ptr, json_len);
    let Ok(s) = std::str::from_utf8(bytes) else {
        return std::ptr::null_mut();
    };
    match Tokenizer::from_json(s) {
        Ok(tok) => Box::into_raw(Box::new(tok)),
        Err(_) => std::ptr::null_mut(),
    }
}

#[no_mangle]
pub unsafe extern "C" fn ignis_tok_free(tok: *mut Tokenizer) {
    if !tok.is_null() {
        drop(Box::from_raw(tok));
    }
}

#[no_mangle]
pub unsafe extern "C" fn ignis_tok_encode(
    tok: *const Tokenizer,
    text_ptr: *const u8,
    text_len: usize,
    out_len_ptr: *mut usize,
) -> *mut u32 {
    let tok = &*tok;
    let bytes = slice::from_raw_parts(text_ptr, text_len);
    let s = std::str::from_utf8_unchecked(bytes);
    let mut ids = tok.encode(s);
    ids.shrink_to_fit();
    *out_len_ptr = ids.len();
    let ptr = ids.as_mut_ptr();
    std::mem::forget(ids);
    ptr
}

#[no_mangle]
pub unsafe extern "C" fn ignis_tok_free_u32(ptr: *mut u32, len: usize) {
    if !ptr.is_null() && len > 0 {
        drop(Vec::from_raw_parts(ptr, len, len));
    }
}

#[no_mangle]
pub unsafe extern "C" fn ignis_tok_decode(
    tok: *const Tokenizer,
    ids_ptr: *const u32,
    ids_len: usize,
    skip_special: u32,
    out_len_ptr: *mut usize,
) -> *mut u8 {
    let tok = &*tok;
    let ids = slice::from_raw_parts(ids_ptr, ids_len);
    let mut s = tok.decode(ids, skip_special != 0).into_bytes();
    s.shrink_to_fit();
    *out_len_ptr = s.len();
    let ptr = s.as_mut_ptr();
    std::mem::forget(s);
    ptr
}

#[no_mangle]
pub unsafe extern "C" fn ignis_tok_free_bytes(ptr: *mut u8, len: usize) {
    if !ptr.is_null() && len > 0 {
        drop(Vec::from_raw_parts(ptr, len, len));
    }
}

#[no_mangle]
pub unsafe extern "C" fn ignis_tok_vocab_size(tok: *const Tokenizer) -> usize {
    if tok.is_null() {
        return 0;
    }
    (&*tok).vocab_size()
}

#[no_mangle]
pub unsafe extern "C" fn ignis_fsm_create(
    tok: *const Tokenizer,
    schema_ptr: *const u8,
    schema_len: usize,
    eos_ptr: *const u32,
    eos_len: usize,
) -> *mut TokenFsm {
    if tok.is_null() {
        return std::ptr::null_mut();
    }
    let tok = &*tok;
    let schema_bytes = if schema_ptr.is_null() || schema_len == 0 {
        &[]
    } else {
        slice::from_raw_parts(schema_ptr, schema_len)
    };
    let Ok(schema_str) = std::str::from_utf8(schema_bytes) else {
        return std::ptr::null_mut();
    };
    let eos_ids = if eos_ptr.is_null() || eos_len == 0 {
        &[][..]
    } else {
        slice::from_raw_parts(eos_ptr, eos_len)
    };
    match tok.create_fsm(schema_str, eos_ids) {
        Ok(fsm) => Box::into_raw(Box::new(fsm)),
        Err(_) => std::ptr::null_mut(),
    }
}

#[no_mangle]
pub unsafe extern "C" fn ignis_fsm_allowed_bitset(
    fsm: *mut TokenFsm,
    out_words_len_ptr: *mut usize,
    out_allowed_count_ptr: *mut u32,
) -> *const u32 {
    if fsm.is_null() {
        return std::ptr::null();
    }
    let fsm = &mut *fsm;
    let (words, count) = fsm.allowed_bitset();
    if !out_words_len_ptr.is_null() {
        *out_words_len_ptr = words.len();
    }
    if !out_allowed_count_ptr.is_null() {
        *out_allowed_count_ptr = count;
    }
    words.as_ptr()
}

#[no_mangle]
pub unsafe extern "C" fn ignis_fsm_mask_logits(
    fsm: *mut TokenFsm,
    logits_ptr: *mut f32,
    vocab_size: usize,
) -> u32 {
    if fsm.is_null() || logits_ptr.is_null() {
        return 0;
    }
    let fsm = &mut *fsm;
    let logits = slice::from_raw_parts_mut(logits_ptr, vocab_size);
    fsm.mask_logits(logits)
}

#[no_mangle]
pub unsafe extern "C" fn ignis_fsm_advance(fsm: *mut TokenFsm, token_id: u32) -> u32 {
    if fsm.is_null() {
        return 0;
    }
    let fsm = &mut *fsm;
    if fsm.advance(token_id) { 1 } else { 0 }
}

#[no_mangle]
pub unsafe extern "C" fn ignis_fsm_is_done(fsm: *const TokenFsm) -> u32 {
    if fsm.is_null() {
        return 0;
    }
    if (&*fsm).is_done() { 1 } else { 0 }
}

#[no_mangle]
pub unsafe extern "C" fn ignis_fsm_reset(fsm: *mut TokenFsm) {
    if !fsm.is_null() {
        (&mut *fsm).reset();
    }
}

#[no_mangle]
pub unsafe extern "C" fn ignis_fsm_free(fsm: *mut TokenFsm) {
    if !fsm.is_null() {
        drop(Box::from_raw(fsm));
    }
}

#[cfg(test)]
mod fsm_tests {
    use super::*;

    fn mini_tokenizer_json() -> String {
        serde_json::json!({
            "added_tokens": [
                { "id": 0, "content": "<eos>", "special": true }
            ],
            "normalizer": null,
            "pre_tokenizer": { "type": "ByteLevel", "add_prefix_space": false, "use_regex": false },
            "decoder": { "type": "ByteLevel" },
            "model": {
                "vocab": {
                    "{": 1, "}": 2, "\"": 3, ":": 4, ",": 5, "Ġ": 6,
                    "\"city\":": 7, "\"country\":": 8, "\"Paris\"": 9, "\"France\"": 10,
                    "\"sentiment\":": 11, "\"score\":": 12, "\"verified\":": 13,
                    "\"positive\"": 14, "true": 15, "false": 16, "42": 17,
                    "invalid_bare": 18, "中": 19, "text": 20, "[": 21, "]": 22
                },
                "merges": []
            }
        }).to_string()
    }

    #[test]
    fn test_token_fsm_step_and_mask() {
        let tok = Tokenizer::from_json(&mini_tokenizer_json()).unwrap();
        let schema = r#"{"type":"object","properties":{"city":{"type":"string"},"country":{"type":"string"}},"required":["city","country"]}"#;
        let mut fsm = tok.create_fsm(schema, &[0]).unwrap();
        let mut logits = vec![0.0f32; tok.vocab_size()];
        let cnt = fsm.mask_logits(&mut logits);
        assert!(cnt >= 1);
        assert!(logits[1].is_finite()); // '{'
        assert!(logits[18].is_infinite() && logits[18] < 0.0); // 'invalid_bare'
        assert!(logits[0].is_infinite() && logits[0] < 0.0); // '<eos>' not allowed at start

        assert!(fsm.advance(1)); // {
        assert!(fsm.advance(7)); // "city":
        assert!(fsm.advance(9)); // "Paris"
        assert!(fsm.advance(5)); // ,
        assert!(fsm.advance(8)); // "country":
        assert!(fsm.advance(10)); // "France"
        assert!(fsm.advance(2)); // }
        assert!(fsm.is_done());

        let mut end_logits = vec![0.0f32; tok.vocab_size()];
        fsm.mask_logits(&mut end_logits);
        assert!(end_logits[0].is_finite()); // <eos> now allowed!
        assert!(fsm.advance(0));
    }
}
