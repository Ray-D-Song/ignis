use std::collections::HashMap;
use std::slice;

use fancy_regex::Regex;
use serde::Deserialize;

#[derive(Deserialize)]
struct TokenizerJson {
    #[serde(default)]
    added_tokens: Vec<AddedTokenJson>,
    pre_tokenizer: Option<PreTokenizerJson>,
    model: ModelJson,
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
}

enum PreStep {
    SplitRegex { regex: Regex, #[allow(dead_code)] invert: bool },
    SplitLiteral { literal: String, #[allow(dead_code)] invert: bool },
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
    merges: HashMap<(u32, u32), (u32, u32)>, // (a, b) -> (rank, merged_id)
    added_tokens: Vec<AddedTokenEntry>,
    added_set: HashMap<u32, String>,
    pre_steps: Vec<PreStep>,
    gpt2_default_re: Option<Regex>,
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

fn compile_pre_steps(pt: &PreTokenizerJson, out: &mut Vec<PreStep>) -> Result<(), String> {
    match pt {
        PreTokenizerJson::Sequence { pretokenizers } => {
            for p in pretokenizers {
                compile_pre_steps(p, out)?;
            }
        }
        PreTokenizerJson::Split { pattern, behavior: _, invert } => {
            match pattern {
                PatternJson::Regex { regex } => {
                    let re = Regex::new(regex).map_err(|e| format!("Regex compile error ({regex}): {e}"))?;
                    out.push(PreStep::SplitRegex { regex: re, invert: *invert });
                }
                PatternJson::String { literal } => {
                    out.push(PreStep::SplitLiteral { literal: literal.clone(), invert: *invert });
                }
            }
        }
        PreTokenizerJson::Digits { individual_digits } => {
            if *individual_digits {
                out.push(PreStep::IndividualDigits);
            }
        }
        PreTokenizerJson::ByteLevel { add_prefix_space, use_regex } => {
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
                added_set.insert(a.id, a.content.clone());
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

        let mut byte_tok_ids = [0u32; 256];
        for b in 0..256 {
            let mut s = String::new();
            s.push(byte_to_char[b]);
            if let Some(&id) = parsed.model.vocab.get(&s) {
                byte_tok_ids[b] = id;
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

        let mut pre_steps = Vec::new();
        if let Some(pt) = &parsed.pre_tokenizer {
            compile_pre_steps(pt, &mut pre_steps)?;
        }

        let gpt2_default_re = Regex::new(
            r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
        ).ok();

        Ok(Self {
            id_to_token,
            byte_to_char,
            byte_tok_ids,
            char_to_byte,
            merges,
            added_tokens,
            added_set,
            pre_steps,
            gpt2_default_re,
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

    fn pretokenize(&self, text: &str) -> Vec<String> {
        let mut pieces = vec![text.to_string()];
        for step in &self.pre_steps {
            let mut next = Vec::with_capacity(pieces.len() * 2);
            match step {
                PreStep::SplitRegex { regex, invert: _ } => {
                    for p in &pieces {
                        Self::apply_regex_isolated(regex, p, &mut next);
                    }
                }
                PreStep::SplitLiteral { literal, invert: _ } => {
                    for p in &pieces {
                        if literal.is_empty() {
                            next.push(p.clone());
                            continue;
                        }
                        let mut last = 0;
                        for (idx, _) in p.match_indices(literal.as_str()) {
                            if idx > last {
                                next.push(p[last..idx].to_string());
                            }
                            next.push(literal.clone());
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
        let bytes = piece.as_bytes();
        if bytes.is_empty() {
            return;
        }
        if bytes.len() == 1 {
            out.push(self.byte_tok_ids[bytes[0] as usize]);
            return;
        }
        let mut ids: Vec<u32> = bytes.iter().map(|&b| self.byte_tok_ids[b as usize]).collect();
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
            if let Some(s) = self.added_set.get(&id) {
                if !skip_special {
                    raw_bytes.extend_from_slice(s.as_bytes());
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
