#[derive(Debug, Clone, Copy)]
pub struct Config {
    pub dim: usize,        // transformer width
    pub hidden_dim: usize, // ffn hidden size
    pub n_layers: usize,   // num of layers
    pub n_heads: usize,    // attention heads
    pub n_kv_heads: usize, // key/value heads
    pub vocab_size: usize, // number of tokens
    pub seq_len: usize,    // max context
}

#[derive(Debug, Clone, Copy)]
pub struct Weights<'a> {
    pub token_embedding: &'a [f32],

    // attention normalization
    pub rms_att: &'a [f32],

    // attention weights
    pub wq: &'a [f32], // query
    pub wk: &'a [f32], // key
    pub wv: &'a [f32], // value
    pub wo: &'a [f32], // output

    pub rms_ffn: &'a [f32],

    pub w1: &'a [f32],
    pub w2: &'a [f32],
    pub w3: &'a [f32],

    // final normalization
    pub rms_final: &'a [f32],

    pub wcls: &'a [f32],
}

// track pos in weights
struct Cursor<'a> {
    data: &'a [f32],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> &'a [f32] {
        let start = self.pos;
        let end = start + n;

        let slice = &self.data[start..end];

        self.pos = end;

        slice
    }
}

// read model settings from the header
pub fn parse_config(bytes: &[u8]) -> Result<(Config, bool), String> {
    if bytes.len() < 28 {
        return Err("File is too small".to_string());
    }

    let mut header = [0i32; 7];

    for i in 0..7 {
        let start = i * 4;
        header[i] = i32::from_le_bytes(bytes[start..start + 4].try_into().unwrap());
    }

    let raw_vocab = header[5];
    let shared_weights = raw_vocab > 0;

    let config = Config {
        dim: header[0] as usize,
        hidden_dim: header[1] as usize,
        n_layers: header[2] as usize,
        n_heads: header[3] as usize,
        n_kv_heads: header[4] as usize,
        vocab_size: raw_vocab.unsigned_abs() as usize,
        seq_len: header[6] as usize,
    };

    Ok((config, shared_weights))
}

fn read_floats(bytes: &[u8]) -> Result<Vec<f32>, String> {
    if bytes.len() % 4 != 0 {
        return Err("Invalid float data length".to_string());
    }

    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect())
}

pub fn load_weights<'a>(cfg: &Config, floats: &'a [f32], shared_weights: bool) -> Weights<'a> {
    let mut cursor = Cursor {
        data: floats,
        pos: 0,
    };

    let l = cfg.n_layers;
    let dim = cfg.dim;
    let hidden = cfg.hidden_dim;

    let head_size = dim / cfg.n_heads;
    let kv_dim = head_size * cfg.n_kv_heads;

    let token_embedding = cursor.take(cfg.vocab_size * dim);

    let rms_att = cursor.take(l * dim);

    let wq = cursor.take(l * dim * dim);
    let wk = cursor.take(l * dim * kv_dim);
    let wv = cursor.take(l * dim * kv_dim);
    let wo = cursor.take(l * dim * dim);

    let rms_ffn = cursor.take(l * dim);

    let w1 = cursor.take(l * hidden * dim);
    let w2 = cursor.take(l * dim * hidden);
    let w3 = cursor.take(l * hidden * dim);

    let rms_final = cursor.take(dim);

    let rope_size = cfg.seq_len * head_size / 2;

    cursor.take(rope_size);
    cursor.take(rope_size);

    let wcls = if shared_weights {
        token_embedding
    } else {
        cursor.take(cfg.vocab_size * dim)
    };

    Weights {
        token_embedding,
        rms_att,
        wq,
        wk,
        wv,
        wo,
        rms_ffn,
        w1,
        w2,
        w3,
        rms_final,
        wcls,
    }
}
