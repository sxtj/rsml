use std::fs;

use crate::weights::parse_config;

mod tensor;
mod weights;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Hello, world!");

    let bytes = fs::read("stories15M.bin")?;

    let (config, shared_weights) = parse_config(&bytes)?;

    println!("{:#?}", config);
    println!("shared weights: {}", shared_weights);

    Ok(())
}
