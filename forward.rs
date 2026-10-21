use crate::tensor::{embed, rmsnorm};
use crate::weights::{Config, Weights};

pub struct RunState {
    //residual stream
    pub x: Vec<f32>,
    pub xb: Vec<f32>,
    pub xb2: Vec<f32>,

    //feed forward
    pub hb: Vec<f32>,
    pub hb2: Vec<f32>,

    //attention
    pub q: Vec<f32>,
    pub att: Vec<f32>,

    //outputs
    pub logits: Vec<f32>,

    //kv cache
    pub key_cache: Vec<f32>,
    pub value_chache: Vec<f32>,
}

impl RunState {
    pub fn new(cfg: &Config) -> Self {
        let dim = cfg.dim;
        let hidden_dim = cfg.hidden_dim;

        assert!(cfg.n_heads > 0);
        asser!(cfg.n_kv_heads > 0);
        assert_eq!(dim % cfg.n_heads, 0);
        assert_eq!(cfg.n_heads % cfg.n_kv_heads, 0);

        let head_size = dim / cfg.n_heads;
        assert_eq!(head_size % 2, 0);

        let kv_dim = head_size * cfg.n_kv_heads;

        let cache_size = cfg.n_layers * cfg.seq_len * kv_dim;

        Self {
            x: vec![0.0; dim],
            xb: vec![0.0; dim],
            xb2: vec![0.0; dim],

            hb: vec![0.0; hidden_dim],
            hb2: vec![0.0; hidden_dim],

            q: vec![0.0; dim],
            att: vec![0.0; cfg.n_heads * cfg.seq_len],

            logits: vec![0.0; cfg.vocab_size],

            key_cache: vec![0.0; cache_size],
            value_cache: vec![0.0; cache_size],
        }
    }
}

pub struct Transformer<'a> {
    pub config: Config,
    pub weights: Weights<'a>,
    pub state: RunState,
}

impl<'a> Transformer<'a> {
    pub fn new(config: Config, weights: Weights<'a>) -> Self {
        let state = RunState::new(&config);

        Self {
            config,
            weights,
            state,
        }
    }

    pub fn load_token_embedding(&mut self, token: usize) {
        assert!(token < self.config.vocab_size);

        embed(
            &mut self.state.x,
            self.weights.token_embedding,
            token,
            self.config.dim,
        );
    }
}

pub fn normalize_for_attention(&mut self, layer: usize) {
    assert!(layer < self.config.n_layers);

    let dim = self.config.dim;

    let start = layer * dim;
    let end = start + dim;

    let rms_weights = &self.weights.rms_Att[start..end];

    rmsnorm(&mut self.state.xb, &self.state.x, rms_weights, 1e-5);
}
