use std::f32;
struct CLIPAttention {
    embed_dim: u16,
    num_heads: u16,
    head_dim: u16,
    scale: f32,
}
impl CLIPAttention {
    fn new() -> CLIPAttention {
        let embed_dim = 768;
        let num_heads = 12;
        let head_dim = embed_dim / num_heads;
        let scale = 1. / (head_dim as f32).sqrt();
        CLIPAttention {
            embed_dim,
            num_heads,
            head_dim,
            scale,
        }
    }
}
struct CLIPEncoderLayer {}
struct CLIPEncoder {}
impl CLIPEncoder {
    fn new() -> CLIPEncoder {
        CLIPEncoder {}
    }
}
struct CLIPTextEmbeddings {}
impl CLIPTextEmbeddings {
    fn new() -> CLIPTextEmbeddings {
        CLIPTextEmbeddings {}
    }
}
pub struct CLIPTextTransformer {
    embeddings: CLIPTextEmbeddings,
}
impl CLIPTextTransformer {
    pub fn new() -> CLIPTextTransformer {
        let embeddings = CLIPTextEmbeddings::new();
        CLIPTextTransformer { embeddings }
    }
    pub fn call() -> () {}
    pub fn predict_on_batch(&self, input: [&Vec<i32>; 2]) -> () {
        println!("Predicting {:?}", input);
    }
}
