#[derive(Debug, Clone, Copy)]
pub struct Config {
    pub dim: usize, //transformer width
    pub hidden_dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub vocab_size: usize,
    pub seq_len: usize, //max context
}

#[derive(Debug, Clone, Copy)]
pub struct Weights<'a> {
    pub token_embedding: &'a [f32],

    pub rms_att: &'a [f32],

    pub wq: &'a [f32],
    pub wk: &'a [f32],
    pub wv: &'a [f32],
    pub wo: &'a [f32],

    pub rms_ffn: &'a [f32],

    pub w1: &'a [f32],
    pub w2: &'a [f32],
    pub w3: &'a [f32],

    pub rms_final: &'a [f32],

    pub wcls: &'a [f32],
}

struct Cursor<'a> {
    data: &'a [f32],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> &'a [f32] {
        let start = self.pos;
        let end = start + n;

        let slice = &self.sata[start..end];

        self.pos = end;

        slice
    }
}

pub fn parse_config(bytes: &[u8]) -> Result<(Config, bool), String> {
    if bytes.len() < 28 {
        return Err("File is too small".to_string());
    }

    let mut header = [0i32; 7];

    for i in 0..7 {
        let start = i * 4;
        headers[i] = i32::from_le_bytes(bytes[start..start + 4].try_into().unwrap());
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

    Ok((Config, shared_weights))
}

fn read_floats(bytes: &[u8]) -> Result<Vec<f32>, String> {
    if bytes.len() % 4 != 0 {
        return Err("Invalid float data length".to_string());
    }

    Ok(Bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect())
}

pub fn load_weights<'a>(cfg: &config, floats: &'a [f32], shared_weights: bool) -> Weights<'a> {
    let mut cursor = Cursor {
        data: floats,
        pos: 0,
    };

    let l = cfg.n_layers;
    let dim = cfg.dim;
    let hidden = cfg.hidden_dim;

    let head_size = dim / cfg.n_heads;
    let kv_dim = cfg.n_kv_heads;

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

    let wcls = if shared_weights {
        token_embedding
    } else {
        cursor.take(cfg.vocab_size * dim)
    };
}
