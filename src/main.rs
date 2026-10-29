use std::fs;

use crate::forward::Transformer;
use crate::weights::{load_weights, parse_config, read_floats};

mod forward;
mod tensor;
mod weights;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Hello, world!");

    let bytes = fs::read("stories15M.bin")?;

    let (config, shared_weights) = parse_config(&bytes)?;
    let floats = read_floats(&bytes[28..])?;
    let weights = load_weights(&config, &floats, shared_weights);

    let mut model = Transformer::new(config, weights);

    model.load_token_embedding(1);
    println!("embedding: {:?}", &model.state.x[..8]);

    model.normalize_for_attention(0);
    println!("normalized: {:?}", &model.state.xb[..8]);

    model.project_query(0);
    println!("query: {:?}", &model.state.q[..8]);

    model.project_key_value(0, 0);
    println!("key: {:?}", &model.state.key_cache[..8]);
    println!("value: {:?}", &model.state.value_cache[..8]);

    println!("{:#?}", config);
    println!("shared weights: {}", shared_weights);

    Ok(())
}
