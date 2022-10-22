use crate::alphas::_ALPHAS_CUMPROD;
use crate::autoencoder_kl::{Decoder, Encoder};
use crate::clip_encoder::CLIPTextTransformer;
use crate::clip_tokenizer::SimpleTokenizer;
use crate::diffusion_model::UNetModel;
use std::path::PathBuf;

use tensorflow::eager::{self, raw_ops, Context, TensorHandle, ToTensorHandle};
use tensorflow::Tensor;

pub struct StableDiffusion {
    ctx: Context,
    img_height: u16,
    img_width: u16,
    tokenizer: SimpleTokenizer,
    text_encoder: CLIPTextTransformer,
    diffusion_model: UNetModel,
    decoder: Decoder,
    encoder: Encoder,
    unconditional_tokens: Tensor<i32>,
}

impl StableDiffusion {
    pub fn new(img_height: u16, img_width: u16) -> StableDiffusion {
        // Create an eager execution context
        let opts = eager::ContextOptions::new();
        let ctx = eager::Context::new(opts).expect("Failed to setup context");
        let tokenizer = SimpleTokenizer::new();
        let (text_encoder, diffusion_model, decoder, encoder) =
            get_models(img_height, img_width, false);
        let mut unconditional_tokens = Vec::from([49406]);
        unconditional_tokens.extend([49407].repeat(76));
        let unconditional_tokens = Tensor::from(unconditional_tokens.as_slice());
        StableDiffusion {
            ctx,
            img_height,
            img_width,
            tokenizer,
            text_encoder,
            diffusion_model,
            decoder,
            encoder,
            unconditional_tokens,
        }
    }

    pub fn generate(
        &self,
        prompt: String,
        num_steps: u8,
        unconditional_guidance_scale: f32,
        temperature: i8,
        seed: Option<u32>,
        input_image: Option<PathBuf>,
        input_mask: Option<PathBuf>,
        input_image_strength: f32,
    ) -> () {
        // Tokenize prompt (i.e. starting context)
        let inputs = self.tokenizer.encode(&prompt);
        assert!(
            inputs.len() < 77,
            "Prompt is too long (should be < 77 tokens)"
        );
        let mut phrase: Vec<i32> = inputs.iter().map(|x| *x as i32).collect();
        phrase.extend([49407].repeat(77 - inputs.len()));

        // Encode prompt tokens (and their positions) into a "context vector"
        let pos_ids = Vec::from_iter(0..77 as i32);
        let context = self.text_encoder.predict_on_batch([&phrase, &pos_ids]);

        // TODO
        // Resize input image
        let input_image_tensor = match input_image {
            Some(ref image) => {
                let ctx = &self.ctx;
                let img_handle = image
                    .as_path()
                    .display()
                    .to_string()
                    .to_handle(ctx)
                    .expect("Failed to get image handle");
                let buf = raw_ops::read_file(ctx, &img_handle).expect("Failed to read file");
                let img = raw_ops::decode_image(ctx, &buf).expect("Failed to decode image");
                let cast_to_float = raw_ops::Cast::new().DstT(tensorflow::DataType::Float);
                let img = cast_to_float
                    .call(ctx, &img)
                    .expect("Failed to cast image to float");
                let batch =
                    raw_ops::expand_dims(ctx, &img, &0).expect("Failed to expand dimensions"); // add batch dim
                let readonly_x = batch.resolve().expect("Failed to resolve batch");
                // The current eager API implementation requires unsafe block to feed the tensor into a graph.
                let x: Tensor<f32> = unsafe { readonly_x.into_tensor() };
                Some(x)
            }
            None => None,
        };
        // TODO
        // Resize input mask
        // Encode uncoditional tokens and their positions into an "unconditional context vector"
        let unconditional_context = self
            .text_encoder
            .predict_on_batch([&self.unconditional_tokens.to_vec(), &pos_ids]);
        let timesteps: Vec<u16> = (1..1000)
            .step_by((1000 / num_steps as u16).into())
            .collect();
        let input_img_noise_t = timesteps[(timesteps.len() as f32 * input_image_strength) as usize];
        let (latent, alphas, alphas_prev) =
            self.get_starting_parameters(timesteps, seed, input_image_tensor, input_img_noise_t);

        println!(
            "prompt: {:?},\nheight: {:?},\nwidth: {:?},\nsteps: {:?},\nscale: {:?},\ntemperature: {:?},\nseed: {:?},\ninput_image: {:?},\ninput_mask: {:?},\ninput_image_strength: {:?}",
            prompt, self.img_height, self.img_width, num_steps, unconditional_guidance_scale, temperature, seed, input_image, input_mask, input_image_strength
        )
    }
    fn get_starting_parameters(
        &self,
        timesteps: Vec<u16>,
        seed: Option<u32>,
        input_image: Option<Tensor<f32>>,
        input_image_noise_t: u16,
    ) -> (TensorHandle, Vec<f32>, Vec<f32>) {
        let new_height = self.img_height / 8;
        let new_width = self.img_width / 8;
        let mut alphas = Vec::new();
        for t in timesteps.iter() {
            alphas.push(_ALPHAS_CUMPROD[*t as usize])
        }
        let mut alphas_prev = vec![1.0];
        alphas_prev.extend(alphas.clone().pop());
        let latent = if input_image.is_some() {
            // TODO does the same as else atm
            raw_ops::random_standard_normal(&self.ctx, &[1, new_height, new_width, 4])
                .expect("Failed to generate latent")
        } else {
            raw_ops::random_standard_normal(&self.ctx, &[1, new_height, new_width, 4])
                .expect("Failed to generate latent")
        };
        (latent, alphas, alphas_prev)
    }
}

fn get_models(
    img_height: u16,
    img_width: u16,
    download_weights: bool,
) -> (CLIPTextTransformer, UNetModel, Decoder, Encoder) {
    let new_height = img_height / 8;
    let new_width = img_width / 8;

    // Create text encoder
    let text_encoder = CLIPTextTransformer::new();
    // Create diffusion UNet
    let diffusion_model = UNetModel::new();
    // Create decoder
    let decoder = Decoder::new();
    // Create encoder
    let encoder = Encoder::new();
    (text_encoder, diffusion_model, decoder, encoder)
}
