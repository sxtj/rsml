use std::fmt::format;

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
    fn take(&mut self, name: &str, n: usize) -> Result<&'a [f32], String> {
        let remaining = self.data.len() - self.pos;

        if n > remaining {
            return Err(format!(
                "{name} expected {n} floats, only {remaining} remain"
            ));
        }

        let start = self.pos;
        self.pos += n;
        Ok(&self.data[start..self.pos])
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

    if header[3] <= 0 || header[4] <= 0 {
        return Err("Head counts must be positive".to_string());
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

    if config.dim % config.n_heads != 0 {
        return Err("Dim must be divisble by n_heads".to_string());
    }

    if config.n_heads % config.n_kv_heads != 0 {
        return Err("n_heads must be divisible by n_kv_heads".to_string());
    }

    Ok((config, shared_weights))
}

pub fn read_floats(bytes: &[u8]) -> Result<Vec<f32>, String> {
    if bytes.len() % 4 != 0 {
        return Err("Invalid float data length".to_string());
    }

    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect())
}

pub fn load_weights<'a>(
    cfg: &Config,
    floats: &'a [f32],
    shared_weights: bool,
) -> Result<Weights<'a>, String> {
    let mut cursor = Cursor {
        data: floats,
        pos: 0,
    };

    let l = cfg.n_layers;
    let dim = cfg.dim;
    let hidden = cfg.hidden_dim;

    let head_size = dim / cfg.n_heads;
    let kv_dim = head_size * cfg.n_kv_heads;

    let token_embedding = cursor.take("token_embedding", cfg.vocab_size * dim)?;

    let rms_att = cursor.take("rms_att", l * dim)?;

    let wq = cursor.take("wq", l * dim * dim)?;
    let wk = cursor.take("wk", l * dim * kv_dim)?;
    let wv = cursor.take("wv", l * dim * kv_dim)?;
    let wo = cursor.take("wo", l * dim * dim)?;

    let rms_ffn = cursor.take("rms_ffn", l * dim)?;

    let w1 = cursor.take("w1", l * hidden * dim)?;
    let w2 = cursor.take("w2", l * dim * hidden)?;
    let w3 = cursor.take("w3", l * hidden * dim)?;

    let rms_final = cursor.take("rms_final", dim)?;

    let rope_size = cfg.seq_len * head_size / 2;

    cursor.take("rope_cos", rope_size)?;
    cursor.take("rope_sin", rope_size)?;

    let wcls = if shared_weights {
        token_embedding
    } else {
        cursor.take("cfg.vocab_size", cfg.vocab_size * dim)?
    };

    Ok(Weights {
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
    })
}
