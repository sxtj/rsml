






use crate::tensor::{embed, matvec, rmsnorm};
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
    pub value_cache: Vec<f32>,
}

impl RunState {
    pub fn new(cfg: &Config) -> Self {
        let dim = cfg.dim;
        let hidden_dim = cfg.hidden_dim;

        assert!(cfg.n_heads > 0);
        assert!(cfg.n_kv_heads > 0);
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

    pub fn normalize_for_attention(&mut self, layer: usize) {
        assert!(layer < self.config.n_layers);

        let dim = self.config.dim;

        let start = layer * dim;
        let end = start + dim;

        let rms_weights = &self.weights.rms_att[start..end];

        rmsnorm(&mut self.state.xb, &self.state.x, rms_weights, 1e-5);
    }

    pub fn project_query(&mut self, layer: usize) {
        assert!(layer < self.config.n_layers);

        let dim = self.config.dim;

        let start = layer * dim * dim;
        let end = layer + dim * dim;

        matvec(
            &mut self.state.q,
            &self.weights.wq[start..end],
            &self.state.xb,
            dim,
            dim,
        );
    }

    pub fn project_key_value(&mut self, layer: usize, pos: usize) {
        assert!(layer < self.config.n_layers);
        assert!(pos < self.config.seq_len);

        let dim = self.config.dim;
        let head_size = dim / self.config.n_heads;
        let kv_dim = head_size * self.config.n_kv_heads;

        let weight_start = layer * kv_dim * dim;
        let weight_end = weight_start + kv_dim * dim;

        let cache_start = (layer * self.config.seq_len + pos) * kv_dim;
        let cache_end = cache_start + kv_dim;

        matvec(
            &mut self.state.key_cache[cache_start..cache_end],
            &self.weights.wk[weight_start..weight_end],
            &self.state.xb,
            kv_dim,
            dim,
        );

        matvec(
            &mut self.state.value_cache[cache_start..cache_end],
            &self.weights.wv[weight_start..weight_end],
            &self.state.xb,
            kv_dim,
            dim,
        );
    }
}
