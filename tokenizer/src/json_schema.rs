use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use serde_json::Value;

pub const DEAD_STATE: u32 = u32::MAX;
const MAX_DFA_STATES: usize = 8192;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ByteSet(pub [u64; 4]);

impl ByteSet {
    pub const fn empty() -> Self {
        Self([0; 4])
    }

    pub fn single(b: u8) -> Self {
        let mut s = Self::empty();
        s.insert(b);
        s
    }

    pub fn range(lo: u8, hi: u8) -> Self {
        let mut s = Self::empty();
        for b in lo..=hi {
            s.insert(b);
        }
        s
    }

    pub fn from_bytes(bytes: &[u8]) -> Self {
        let mut s = Self::empty();
        for &b in bytes {
            s.insert(b);
        }
        s
    }

    pub fn insert(&mut self, b: u8) {
        self.0[(b >> 6) as usize] |= 1u64 << (b & 63);
    }

    pub fn remove(&mut self, b: u8) {
        self.0[(b >> 6) as usize] &= !(1u64 << (b & 63));
    }

    pub fn contains(&self, b: u8) -> bool {
        (self.0[(b >> 6) as usize] & (1u64 << (b & 63))) != 0
    }

    pub fn union(&self, other: &Self) -> Self {
        Self([
            self.0[0] | other.0[0],
            self.0[1] | other.0[1],
            self.0[2] | other.0[2],
            self.0[3] | other.0[3],
        ])
    }
}

struct NfaState {
    eps: Vec<usize>,
    edges: Vec<(ByteSet, usize)>,
}

struct Nfa {
    states: Vec<NfaState>,
    start: usize,
    accept: usize,
}

impl Nfa {
    fn new() -> Self {
        Self {
            states: Vec::new(),
            start: 0,
            accept: 0,
        }
    }

    fn new_state(&mut self) -> usize {
        let id = self.states.len();
        self.states.push(NfaState {
            eps: Vec::new(),
            edges: Vec::new(),
        });
        id
    }

    fn add_eps(&mut self, from: usize, to: usize) {
        self.states[from].eps.push(to);
    }

    fn add_edge(&mut self, from: usize, set: ByteSet, to: usize) {
        self.states[from].edges.push((set, to));
    }

    fn emit_literal(&mut self, s: &str, start: usize, end: usize) {
        let bytes = s.as_bytes();
        if bytes.is_empty() {
            self.add_eps(start, end);
            return;
        }
        let mut cur = start;
        for (idx, &b) in bytes.iter().enumerate() {
            let next = if idx + 1 == bytes.len() { end } else { self.new_state() };
            self.add_edge(cur, ByteSet::single(b), next);
            cur = next;
        }
    }

    fn emit_star(&mut self, start: usize, end: usize, body: impl Fn(&mut Nfa, usize, usize)) {
        let split = self.new_state();
        let body_start = self.new_state();
        let body_end = self.new_state();
        self.add_eps(start, split);
        self.add_eps(split, body_start);
        self.add_eps(split, end);
        body(self, body_start, body_end);
        self.add_eps(body_end, split);
    }

    fn emit_plus(&mut self, start: usize, end: usize, body: impl Fn(&mut Nfa, usize, usize)) {
        let mid = self.new_state();
        body(self, start, mid);
        self.emit_star(mid, end, body);
    }

    fn emit_opt(&mut self, start: usize, end: usize, body: impl FnOnce(&mut Nfa, usize, usize)) {
        self.add_eps(start, end);
        let b0 = self.new_state();
        self.add_eps(start, b0);
        body(self, b0, end);
    }

    fn emit_ws(&mut self, start: usize, end: usize) {
        let ws = ByteSet::from_bytes(b" \t\r\n");
        self.emit_star(start, end, |nfa, a, b| {
            nfa.add_edge(a, ws, b);
        });
    }

    fn emit_string(&mut self, start: usize, end: usize) {
        let q = ByteSet::single(b'"');
        let s1 = self.new_state();
        let body0 = self.new_state();
        let body1 = self.new_state();
        self.add_edge(start, q, s1);
        self.add_eps(s1, body0);

        let mut safe_ascii = ByteSet::range(0x20, 0x7E);
        safe_ascii.remove(b'"');
        safe_ascii.remove(b'\\');
        let simple_esc = ByteSet::from_bytes(b"\"\\/bfnrt");
        let hex_digit = ByteSet::range(b'0', b'9')
            .union(&ByteSet::range(b'a', b'f'))
            .union(&ByteSet::range(b'A', b'F'));
        let cont = ByteSet::range(0x80, 0xBF);

        self.emit_star(body0, body1, |nfa, a, b| {
            // Safe printable ASCII
            nfa.add_edge(a, safe_ascii, b);

            // JSON escape sequence: \("|\|/|b|f|n|r|t|uXXXX)
            let esc = nfa.new_state();
            nfa.add_edge(a, ByteSet::single(b'\\'), esc);
            nfa.add_edge(esc, simple_esc, b);
            let u0 = nfa.new_state();
            let u1 = nfa.new_state();
            let u2 = nfa.new_state();
            let u3 = nfa.new_state();
            nfa.add_edge(esc, ByteSet::single(b'u'), u0);
            nfa.add_edge(u0, hex_digit, u1);
            nfa.add_edge(u1, hex_digit, u2);
            nfa.add_edge(u2, hex_digit, u3);
            nfa.add_edge(u3, hex_digit, b);

            // 2-byte UTF-8
            let m2 = nfa.new_state();
            nfa.add_edge(a, ByteSet::range(0xC2, 0xDF), m2);
            nfa.add_edge(m2, cont, b);

            // 3-byte UTF-8
            let m3a = nfa.new_state();
            let m3b = nfa.new_state();
            let m3c = nfa.new_state();
            let m3_tail = nfa.new_state();
            nfa.add_edge(a, ByteSet::single(0xE0), m3a);
            nfa.add_edge(m3a, ByteSet::range(0xA0, 0xBF), m3_tail);
            nfa.add_edge(
                a,
                ByteSet::range(0xE1, 0xEC).union(&ByteSet::range(0xEE, 0xEF)),
                m3b,
            );
            nfa.add_edge(m3b, cont, m3_tail);
            nfa.add_edge(a, ByteSet::single(0xED), m3c);
            nfa.add_edge(m3c, ByteSet::range(0x80, 0x9F), m3_tail);
            nfa.add_edge(m3_tail, cont, b);

            // 4-byte UTF-8
            let m4a = nfa.new_state();
            let m4b = nfa.new_state();
            let m4c = nfa.new_state();
            let m4_mid = nfa.new_state();
            let m4_tail = nfa.new_state();
            nfa.add_edge(a, ByteSet::single(0xF0), m4a);
            nfa.add_edge(m4a, ByteSet::range(0x90, 0xBF), m4_mid);
            nfa.add_edge(a, ByteSet::range(0xF1, 0xF3), m4b);
            nfa.add_edge(m4b, cont, m4_mid);
            nfa.add_edge(a, ByteSet::single(0xF4), m4c);
            nfa.add_edge(m4c, ByteSet::range(0x80, 0x8F), m4_mid);
            nfa.add_edge(m4_mid, cont, m4_tail);
            nfa.add_edge(m4_tail, cont, b);
        });

        self.add_edge(body1, q, end);
    }

    fn emit_integer(&mut self, start: usize, end: usize) {
        let sign_end = self.new_state();
        self.emit_opt(start, sign_end, |nfa, a, b| {
            nfa.add_edge(a, ByteSet::single(b'-'), b);
        });
        // Zero branch
        self.add_edge(sign_end, ByteSet::single(b'0'), end);
        // [1-9][0-9]*
        let d1 = self.new_state();
        self.add_edge(sign_end, ByteSet::range(b'1', b'9'), d1);
        self.emit_star(d1, end, |nfa, a, b| {
            nfa.add_edge(a, ByteSet::range(b'0', b'9'), b);
        });
    }

    fn emit_number(&mut self, start: usize, end: usize) {
        let int_end = self.new_state();
        self.emit_integer(start, int_end);

        let frac_end = self.new_state();
        self.emit_opt(int_end, frac_end, |nfa, a, b| {
            let dot = nfa.new_state();
            nfa.add_edge(a, ByteSet::single(b'.'), dot);
            nfa.emit_plus(dot, b, |nfa2, x, y| {
                nfa2.add_edge(x, ByteSet::range(b'0', b'9'), y);
            });
        });

        self.emit_opt(frac_end, end, |nfa, a, b| {
            let e0 = nfa.new_state();
            let s0 = nfa.new_state();
            nfa.add_edge(a, ByteSet::from_bytes(b"eE"), e0);
            nfa.emit_opt(e0, s0, |nfa2, x, y| {
                nfa2.add_edge(x, ByteSet::from_bytes(b"+-"), y);
            });
            nfa.emit_plus(s0, b, |nfa2, x, y| {
                nfa2.add_edge(x, ByteSet::range(b'0', b'9'), y);
            });
        });
    }

    fn emit_boolean(&mut self, start: usize, end: usize) {
        let t0 = self.new_state();
        let f0 = self.new_state();
        self.add_eps(start, t0);
        self.add_eps(start, f0);
        self.emit_literal("true", t0, end);
        self.emit_literal("false", f0, end);
    }

    fn emit_null(&mut self, start: usize, end: usize) {
        self.emit_literal("null", start, end);
    }

    fn emit_generic_value(&mut self, depth: usize, start: usize, end: usize) {
        let p_str = self.new_state();
        let p_num = self.new_state();
        let p_bool = self.new_state();
        let p_null = self.new_state();
        self.add_eps(start, p_str);
        self.add_eps(start, p_num);
        self.add_eps(start, p_bool);
        self.add_eps(start, p_null);
        self.emit_string(p_str, end);
        self.emit_number(p_num, end);
        self.emit_boolean(p_bool, end);
        self.emit_null(p_null, end);
        if depth > 0 {
            let p_arr = self.new_state();
            let p_obj = self.new_state();
            self.add_eps(start, p_arr);
            self.add_eps(start, p_obj);
            self.emit_generic_array(depth - 1, p_arr, end);
            self.emit_generic_object(depth - 1, p_obj, end);
        }
    }

    fn emit_generic_array(&mut self, depth: usize, start: usize, end: usize) {
        let lb = self.new_state();
        self.add_edge(start, ByteSet::single(b'['), lb);
        let ws0 = self.new_state();
        self.emit_ws(lb, ws0);
        let rb = self.new_state();
        self.add_eps(ws0, rb); // empty []
        let first_val_done = self.new_state();
        self.emit_generic_value(depth, ws0, first_val_done);
        let loop_end = self.new_state();
        self.emit_star(first_val_done, loop_end, |nfa, a, b| {
            let w0 = nfa.new_state();
            let c0 = nfa.new_state();
            let w1 = nfa.new_state();
            nfa.emit_ws(a, w0);
            nfa.add_edge(w0, ByteSet::single(b','), c0);
            nfa.emit_ws(c0, w1);
            nfa.emit_generic_value(depth, w1, b);
        });
        self.emit_ws(loop_end, rb);
        self.add_edge(rb, ByteSet::single(b']'), end);
    }

    fn emit_generic_object(&mut self, depth: usize, start: usize, end: usize) {
        let lb = self.new_state();
        self.add_edge(start, ByteSet::single(b'{'), lb);
        let ws0 = self.new_state();
        self.emit_ws(lb, ws0);
        let rb = self.new_state();
        self.add_eps(ws0, rb); // empty {}

        let pair_start = self.new_state();
        self.add_eps(ws0, pair_start);
        let emit_pair = |nfa: &mut Nfa, a: usize, b: usize| {
            let k_done = nfa.new_state();
            let w0 = nfa.new_state();
            let col = nfa.new_state();
            let w1 = nfa.new_state();
            nfa.emit_string(a, k_done);
            nfa.emit_ws(k_done, w0);
            nfa.add_edge(w0, ByteSet::single(b':'), col);
            nfa.emit_ws(col, w1);
            nfa.emit_generic_value(depth, w1, b);
        };
        let p0_end = self.new_state();
        emit_pair(self, pair_start, p0_end);
        let tail_end = self.new_state();
        self.emit_star(p0_end, tail_end, |nfa, a, b| {
            let w0 = nfa.new_state();
            let c0 = nfa.new_state();
            let w1 = nfa.new_state();
            nfa.emit_ws(a, w0);
            nfa.add_edge(w0, ByteSet::single(b','), c0);
            nfa.emit_ws(c0, w1);
            emit_pair(nfa, w1, b);
        });
        self.emit_ws(tail_end, rb);
        self.add_edge(rb, ByteSet::single(b'}'), end);
    }

    fn emit_schema(&mut self, schema: &Value, start: usize, end: usize) -> Result<(), String> {
        let Some(obj) = schema.as_object() else {
            return Err("JSON schema must be an object".to_string());
        };

        if let Some(const_v) = obj.get("const") {
            let lit = serde_json::to_string(const_v).map_err(|e| e.to_string())?;
            self.emit_literal(&lit, start, end);
            return Ok(());
        }

        if let Some(Value::Array(items)) = obj.get("enum") {
            if items.is_empty() {
                return Err("enum schema cannot be empty".to_string());
            }
            for v in items {
                let lit = serde_json::to_string(v).map_err(|e| e.to_string())?;
                let br = self.new_state();
                self.add_eps(start, br);
                self.emit_literal(&lit, br, end);
            }
            return Ok(());
        }

        for kw in ["anyOf", "oneOf"] {
            if let Some(Value::Array(alts)) = obj.get(kw) {
                if alts.is_empty() {
                    return Err(format!("{kw} cannot be empty"));
                }
                for sub in alts {
                    let br = self.new_state();
                    self.add_eps(start, br);
                    self.emit_schema(sub, br, end)?;
                }
                return Ok(());
            }
        }

        if let Some(Value::Array(types)) = obj.get("type") {
            if types.is_empty() {
                return Err("type array cannot be empty".to_string());
            }
            for t in types {
                let mut cloned = obj.clone();
                cloned.insert("type".to_string(), t.clone());
                let br = self.new_state();
                self.add_eps(start, br);
                self.emit_schema(&Value::Object(cloned), br, end)?;
            }
            return Ok(());
        }

        let ty = obj.get("type").and_then(|v| v.as_str()).unwrap_or_else(|| {
            if obj.contains_key("properties") {
                "object"
            } else if obj.contains_key("items") {
                "array"
            } else {
                "object"
            }
        });

        match ty {
            "string" => self.emit_string(start, end),
            "integer" => self.emit_integer(start, end),
            "number" => self.emit_number(start, end),
            "boolean" => self.emit_boolean(start, end),
            "null" => self.emit_null(start, end),
            "array" => self.emit_array_schema(obj, start, end)?,
            "object" => self.emit_object_schema(obj, start, end)?,
            other => return Err(format!("Unsupported schema type: {other}")),
        }
        Ok(())
    }

    fn emit_array_schema(
        &mut self,
        obj: &serde_json::Map<String, Value>,
        start: usize,
        end: usize,
    ) -> Result<(), String> {
        let min_items = obj.get("minItems").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        let max_items = obj.get("maxItems").and_then(|v| v.as_u64()).map(|v| (v as usize).min(12));
        let item_schema = obj.get("items");

        let lb = self.new_state();
        self.add_edge(start, ByteSet::single(b'['), lb);
        let after_lb = self.new_state();
        self.emit_ws(lb, after_lb);

        let emit_item = |nfa: &mut Nfa, a: usize, b: usize| -> Result<(), String> {
            match item_schema {
                Some(s) => nfa.emit_schema(s, a, b),
                None => {
                    nfa.emit_generic_value(1, a, b);
                    Ok(())
                }
            }
        };

        let before_rb = self.new_state();
        if let Some(max_n) = max_items {
            if max_n < min_items {
                return Err("array maxItems < minItems".to_string());
            }
            if min_items == 0 {
                self.add_eps(after_lb, before_rb);
            }
            if max_n >= 1 {
                let mut cur = self.new_state();
                emit_item(self, after_lb, cur)?;
                if min_items <= 1 {
                    self.add_eps(cur, before_rb);
                }
                for count in 2..=max_n {
                    let w0 = self.new_state();
                    let c0 = self.new_state();
                    let w1 = self.new_state();
                    let next = self.new_state();
                    self.emit_ws(cur, w0);
                    self.add_edge(w0, ByteSet::single(b','), c0);
                    self.emit_ws(c0, w1);
                    emit_item(self, w1, next)?;
                    if count >= min_items {
                        self.add_eps(next, before_rb);
                    }
                    cur = next;
                }
            }
        } else {
            if min_items == 0 {
                self.add_eps(after_lb, before_rb);
            }
            let mut cur = self.new_state();
            emit_item(self, after_lb, cur)?;
            for _ in 1..min_items {
                let w0 = self.new_state();
                let c0 = self.new_state();
                let w1 = self.new_state();
                let next = self.new_state();
                self.emit_ws(cur, w0);
                self.add_edge(w0, ByteSet::single(b','), c0);
                self.emit_ws(c0, w1);
                emit_item(self, w1, next)?;
                cur = next;
            }
            // Zero or more trailing `, item`
            let split = self.new_state();
            let tail_body = self.new_state();
            let tail_back = self.new_state();
            self.add_eps(cur, split);
            self.add_eps(split, before_rb);
            self.add_eps(split, tail_body);
            let w0 = self.new_state();
            let c0 = self.new_state();
            let w1 = self.new_state();
            self.emit_ws(tail_body, w0);
            self.add_edge(w0, ByteSet::single(b','), c0);
            self.emit_ws(c0, w1);
            emit_item(self, w1, tail_back)?;
            self.add_eps(tail_back, split);
        }

        let rb = self.new_state();
        self.emit_ws(before_rb, rb);
        self.add_edge(rb, ByteSet::single(b']'), end);
        Ok(())
    }

    fn emit_object_schema(
        &mut self,
        obj: &serde_json::Map<String, Value>,
        start: usize,
        end: usize,
    ) -> Result<(), String> {
        let Some(Value::Object(props_map)) = obj.get("properties") else {
            self.emit_generic_object(2, start, end);
            return Ok(());
        };
        if props_map.is_empty() {
            let lb = self.new_state();
            let rb = self.new_state();
            self.add_edge(start, ByteSet::single(b'{'), lb);
            self.emit_ws(lb, rb);
            self.add_edge(rb, ByteSet::single(b'}'), end);
            return Ok(());
        }

        let mut required_names: Vec<String> = Vec::new();
        let mut required_set = HashSet::new();
        if let Some(Value::Array(req)) = obj.get("required") {
            for r in req {
                if let Some(name) = r.as_str() {
                    if props_map.contains_key(name) && required_set.insert(name.to_string()) {
                        required_names.push(name.to_string());
                    }
                }
            }
        }

        // Order required fields in declaration order of `required`, then remaining optional fields sorted.
        let mut props: Vec<(String, &Value, bool)> = Vec::with_capacity(props_map.len());
        for r in &required_names {
            props.push((r.clone(), &props_map[r], true));
        }
        for (k, v) in props_map {
            if !required_set.contains(k) {
                props.push((k.clone(), v, false));
            }
        }

        let lb = self.new_state();
        self.add_edge(start, ByteSet::single(b'{'), lb);
        let after_lb = self.new_state();
        self.emit_ws(lb, after_lb);

        let before_rb = self.new_state();
        let emit_prop = |nfa: &mut Nfa, name: &str, schema_val: &Value, a: usize, b: usize| -> Result<(), String> {
            let key_lit = serde_json::to_string(name).map_err(|e| e.to_string())?;
            let key_done = nfa.new_state();
            let w0 = nfa.new_state();
            let col = nfa.new_state();
            let w1 = nfa.new_state();
            nfa.emit_literal(&key_lit, a, key_done);
            nfa.emit_ws(key_done, w0);
            nfa.add_edge(w0, ByteSet::single(b':'), col);
            nfa.emit_ws(col, w1);
            nfa.emit_schema(schema_val, w1, b)
        };

        let emit_comma = |nfa: &mut Nfa, a: usize, b: usize| {
            let w0 = nfa.new_state();
            let c0 = nfa.new_state();
            nfa.emit_ws(a, w0);
            nfa.add_edge(w0, ByteSet::single(b','), c0);
            nfa.emit_ws(c0, b);
        };

        let n = props.len();
        if n <= 3 {
            // Subset DAG allowing any field ordering for 1..3 properties
            let req_mask: usize = props
                .iter()
                .enumerate()
                .filter(|(_, (_, _, req))| *req)
                .fold(0, |acc, (i, _)| acc | (1 << i));
            let num_masks = 1usize << n;
            let mut mask_nodes = Vec::with_capacity(num_masks);
            for _ in 0..num_masks {
                mask_nodes.push(self.new_state());
            }
            self.add_eps(after_lb, mask_nodes[0]);
            for mask in 0..num_masks {
                if (mask & req_mask) == req_mask {
                    self.add_eps(mask_nodes[mask], before_rb);
                }
                for (i, (k, v, _)) in props.iter().enumerate() {
                    if (mask & (1 << i)) != 0 {
                        continue;
                    }
                    let next_mask = mask | (1 << i);
                    let prop_start = if mask == 0 {
                        mask_nodes[mask]
                    } else {
                        let ps = self.new_state();
                        emit_comma(self, mask_nodes[mask], ps);
                        ps
                    };
                    emit_prop(self, k, v, prop_start, mask_nodes[next_mask])?;
                }
            }
        } else {
            // Linear O(N) chain `(idx, emitted_any)` for schemas with >= 4 fields
            let mut nodes_empty = Vec::with_capacity(n + 1);
            let mut nodes_some = Vec::with_capacity(n + 1);
            for _ in 0..=n {
                nodes_empty.push(self.new_state());
                nodes_some.push(self.new_state());
            }
            self.add_eps(after_lb, nodes_empty[0]);
            for (i, (k, v, req)) in props.iter().enumerate() {
                // From empty -> emit prop[i] -> some[i+1]
                emit_prop(self, k, v, nodes_empty[i], nodes_some[i + 1])?;
                // From some -> comma + prop[i] -> some[i+1]
                let after_comma = self.new_state();
                emit_comma(self, nodes_some[i], after_comma);
                emit_prop(self, k, v, after_comma, nodes_some[i + 1])?;
                if !*req {
                    self.add_eps(nodes_empty[i], nodes_empty[i + 1]);
                    self.add_eps(nodes_some[i], nodes_some[i + 1]);
                }
            }
            if required_names.is_empty() {
                self.add_eps(nodes_empty[n], before_rb);
            }
            self.add_eps(nodes_some[n], before_rb);
        }

        let rb = self.new_state();
        self.emit_ws(before_rb, rb);
        self.add_edge(rb, ByteSet::single(b'}'), end);
        Ok(())
    }
}

pub struct Dfa {
    pub num_states: usize,
    pub num_classes: usize,
    pub byte_to_class: [u8; 256],
    pub class_rep_byte: Vec<u8>,
    pub transitions: Vec<u32>,
    pub accepting: Vec<bool>,
}

impl Dfa {
    pub fn from_json_schema_str(schema_json: &str) -> Result<Self, String> {
        let trimmed = schema_json.trim();
        let val: Value = if trimmed.is_empty() {
            serde_json::json!({ "type": "object" })
        } else {
            serde_json::from_str(trimmed).map_err(|e| format!("Schema JSON error: {e}"))?
        };
        Self::from_json_schema(&val)
    }

    pub fn from_json_schema(schema: &Value) -> Result<Self, String> {
        let mut nfa = Nfa::new();
        let s0 = nfa.new_state();
        let s1 = nfa.new_state();
        let s2 = nfa.new_state();
        nfa.start = s0;
        nfa.accept = s2;
        nfa.emit_ws(s0, s1);
        let core_end = nfa.new_state();
        nfa.emit_schema(schema, s1, core_end)?;
        nfa.emit_opt(core_end, s2, |nfa2, a, b| {
            nfa2.add_edge(a, ByteSet::from_bytes(b" \t\r\n"), b);
        });
        Self::determinize(&nfa)
    }

    fn compute_byte_classes(nfa: &Nfa) -> ([u8; 256], Vec<u8>) {
        let mut class_of = [0u8; 256];
        let mut num_classes = 1usize;
        let mut unique_sets: Vec<ByteSet> = Vec::new();
        let mut seen = HashSet::new();
        for st in &nfa.states {
            for (bs, _) in &st.edges {
                if seen.insert(*bs) {
                    unique_sets.push(*bs);
                }
            }
        }
        for bs in &unique_sets {
            let mut remap: HashMap<(u8, bool), u8> = HashMap::new();
            let mut next_classes = 0usize;
            for b in 0..=255u8 {
                let key = (class_of[b as usize], bs.contains(b));
                let new_c = *remap.entry(key).or_insert_with(|| {
                    let id = next_classes as u8;
                    next_classes += 1;
                    id
                });
                class_of[b as usize] = new_c;
            }
            num_classes = next_classes;
            if num_classes == 256 {
                break;
            }
        }
        let mut rep = vec![0u8; num_classes];
        let mut filled = vec![false; num_classes];
        for b in 0..=255u8 {
            let c = class_of[b as usize] as usize;
            if !filled[c] {
                filled[c] = true;
                rep[c] = b;
            }
        }
        (class_of, rep)
    }

    fn eps_closure(nfa: &Nfa, seeds: &[usize]) -> Vec<usize> {
        let mut visited = vec![false; nfa.states.len()];
        let mut stack: Vec<usize> = Vec::with_capacity(seeds.len() * 2);
        let mut out: Vec<usize> = Vec::with_capacity(seeds.len() * 2);
        for &s in seeds {
            if !visited[s] {
                visited[s] = true;
                stack.push(s);
            }
        }
        while let Some(u) = stack.pop() {
            out.push(u);
            for &v in &nfa.states[u].eps {
                if !visited[v] {
                    visited[v] = true;
                    stack.push(v);
                }
            }
        }
        out.sort_unstable();
        out
    }

    fn determinize(nfa: &Nfa) -> Result<Self, String> {
        let (byte_to_class, class_rep_byte) = Self::compute_byte_classes(nfa);
        let num_classes = class_rep_byte.len();

        let start_set = Self::eps_closure(nfa, &[nfa.start]);
        let mut state_map: HashMap<Vec<usize>, u32> = HashMap::new();
        let mut queue: VecDeque<(u32, Vec<usize>)> = VecDeque::new();
        let mut transitions: Vec<u32> = Vec::new();
        let mut accepting: Vec<bool> = Vec::new();

        state_map.insert(start_set.clone(), 0);
        accepting.push(start_set.binary_search(&nfa.accept).is_ok());
        transitions.resize(num_classes, DEAD_STATE);
        queue.push_back((0, start_set));

        let mut class_targets: Vec<Vec<usize>> = vec![Vec::new(); num_classes];
        while let Some((sid, nfa_set)) = queue.pop_front() {
            for t in &mut class_targets {
                t.clear();
            }
            for &ns in &nfa_set {
                for &(bs, target) in &nfa.states[ns].edges {
                    for (c, &rep_b) in class_rep_byte.iter().enumerate() {
                        if bs.contains(rep_b) {
                            class_targets[c].push(target);
                        }
                    }
                }
            }
            for c in 0..num_classes {
                if class_targets[c].is_empty() {
                    continue;
                }
                let next_set = Self::eps_closure(nfa, &class_targets[c]);
                let next_id = if let Some(&existing) = state_map.get(&next_set) {
                    existing
                } else {
                    let nid = state_map.len();
                    if nid >= MAX_DFA_STATES {
                        return Err(format!("DFA exceeded {MAX_DFA_STATES} state cap"));
                    }
                    let nid_u32 = nid as u32;
                    accepting.push(next_set.binary_search(&nfa.accept).is_ok());
                    transitions.resize((nid + 1) * num_classes, DEAD_STATE);
                    state_map.insert(next_set.clone(), nid_u32);
                    queue.push_back((nid_u32, next_set));
                    nid_u32
                };
                transitions[sid as usize * num_classes + c] = next_id;
            }
        }

        Self::prune_dead_states(accepting.len(), num_classes, byte_to_class, class_rep_byte, transitions, accepting)
    }

    fn prune_dead_states(
        raw_states: usize,
        num_classes: usize,
        byte_to_class: [u8; 256],
        class_rep_byte: Vec<u8>,
        transitions: Vec<u32>,
        accepting: Vec<bool>,
    ) -> Result<Self, String> {
        let mut rev: Vec<Vec<u32>> = vec![Vec::new(); raw_states];
        for s in 0..raw_states {
            let row = &transitions[s * num_classes..(s + 1) * num_classes];
            for &nxt in row {
                if nxt != DEAD_STATE {
                    rev[nxt as usize].push(s as u32);
                }
            }
        }

        let mut alive = vec![false; raw_states];
        let mut bfs = VecDeque::new();
        for (s, &acc) in accepting.iter().enumerate() {
            if acc {
                alive[s] = true;
                bfs.push_back(s as u32);
            }
        }
        while let Some(u) = bfs.pop_front() {
            for &p in &rev[u as usize] {
                if !alive[p as usize] {
                    alive[p as usize] = true;
                    bfs.push_back(p);
                }
            }
        }

        if !alive[0] {
            return Err("JSON schema matches no valid outputs".to_string());
        }

        let mut remap = vec![DEAD_STATE; raw_states];
        let mut live_count = 0usize;
        for s in 0..raw_states {
            if alive[s] {
                remap[s] = live_count as u32;
                live_count += 1;
            }
        }

        let mut compact_trans = vec![DEAD_STATE; live_count * num_classes];
        let mut compact_acc = vec![false; live_count];
        for s in 0..raw_states {
            let new_s = remap[s];
            if new_s == DEAD_STATE {
                continue;
            }
            let ns = new_s as usize;
            compact_acc[ns] = accepting[s];
            for c in 0..num_classes {
                let raw_t = transitions[s * num_classes + c];
                compact_trans[ns * num_classes + c] = if raw_t == DEAD_STATE {
                    DEAD_STATE
                } else {
                    remap[raw_t as usize]
                };
            }
        }

        // Verify map completeness on BTreeMap import doesn't warn unused
        let _ = BTreeMap::<(), ()>::new();

        Ok(Self {
            num_states: live_count,
            num_classes,
            byte_to_class,
            class_rep_byte,
            transitions: compact_trans,
            accepting: compact_acc,
        })
    }

    #[inline(always)]
    pub fn step_byte(&self, state: u32, byte: u8) -> u32 {
        if state == DEAD_STATE {
            return DEAD_STATE;
        }
        let c = self.byte_to_class[byte as usize] as usize;
        self.transitions[state as usize * self.num_classes + c]
    }

    #[inline(always)]
    pub fn step_bytes(&self, mut state: u32, bytes: &[u8]) -> u32 {
        for &b in bytes {
            state = self.step_byte(state, b);
            if state == DEAD_STATE {
                return DEAD_STATE;
            }
        }
        state
    }

    #[inline(always)]
    pub fn is_accepting(&self, state: u32) -> bool {
        state != DEAD_STATE && self.accepting[state as usize]
    }

    pub fn matches_exact(&self, s: &str) -> bool {
        let end = self.step_bytes(0, s.as_bytes());
        self.is_accepting(end)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flat_object_any_order() {
        let dfa = Dfa::from_json_schema_str(
            r#"{"type":"object","properties":{"city":{"type":"string"},"country":{"type":"string"}},"required":["city","country"]}"#,
        ).unwrap();
        assert!(dfa.matches_exact(r#"{"city": "Paris", "country": "France"}"#));
        assert!(dfa.matches_exact(r#"{"country": "France", "city": "Paris"}"#));
        assert!(!dfa.matches_exact(r#"{"city": "Paris"}"#));
        assert!(!dfa.matches_exact(r#"{"city": "Paris", "extra": 1}"#));
    }

    #[test]
    fn test_enum_integer_bool_and_optional() {
        let dfa = Dfa::from_json_schema_str(
            r#"{
                "type":"object",
                "properties":{
                    "sentiment":{"enum":["positive","negative","neutral"]},
                    "score":{"type":"integer"},
                    "verified":{"type":"boolean"},
                    "note":{"type":"string"}
                },
                "required":["sentiment","score","verified"]
            }"#,
        ).unwrap();
        assert!(dfa.matches_exact(r#"{"sentiment":"positive","score":95,"verified":true}"#));
        assert!(dfa.matches_exact(r#"{"sentiment":"negative","score":-3,"verified":false,"note":"ok"}"#));
        assert!(!dfa.matches_exact(r#"{"sentiment":"happy","score":95,"verified":true}"#));
        assert!(!dfa.matches_exact(r#"{"sentiment":"positive","score":9.5,"verified":true}"#));
    }

    #[test]
    fn test_nested_object_array_utf8_and_dead_state() {
        let dfa = Dfa::from_json_schema_str(
            r#"{
                "type":"object",
                "properties":{
                    "user":{
                        "type":"object",
                        "properties":{"name":{"type":"string"},"age":{"type":"integer"}},
                        "required":["name","age"]
                    },
                    "tags":{"type":"array","items":{"type":"string"},"minItems":1}
                },
                "required":["user","tags"]
            }"#,
        ).unwrap();
        assert!(dfa.matches_exact(r#"{"user": {"name": "Alice 中文", "age": 28}, "tags": ["rust", "wasm"]}"#));
        assert!(dfa.matches_exact(r#"{"tags": ["a"], "user": {"age": 0, "name": "Bob"}}"#));
        assert!(!dfa.matches_exact(r#"{"user": {"name": "Alice", "age": 28}, "tags": []}"#));
        // Dead-state pruning: after `"tru` prefix expecting `true` inside verified bool, `'z'` must hit DEAD_STATE
        let b_dfa = Dfa::from_json_schema_str(r#"{"type":"boolean"}"#).unwrap();
        let st = b_dfa.step_bytes(0, b"tru");
        assert_ne!(st, DEAD_STATE);
        assert_eq!(b_dfa.step_byte(st, b'z'), DEAD_STATE);
    }

    #[test]
    fn test_generic_json_object() {
        let dfa = Dfa::from_json_schema_str(r#"{"type":"object"}"#).unwrap();
        assert!(dfa.matches_exact(r#"{"k": [1, -2.5, true, false, null, {"nested": "ok"}]}"#));
        assert!(dfa.matches_exact(r#"{}"#));
        assert!(!dfa.matches_exact(r#"[1, 2]"#));
    }
}
