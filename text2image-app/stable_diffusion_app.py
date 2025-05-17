import gradio as gr
import torch
from diffusers import StableDiffusionPipeline
from peft import PeftModel
from PIL import Image
import os

# Load the fine-tuned model
def load_model():
    base_model = "CompVis/stable-diffusion-v1-4"
    pipe = StableDiffusionPipeline.from_pretrained(base_model, torch_dtype=torch.float32)
    
    # Load the LoRA weights (updated for safetensors)
    pipe.unet = PeftModel.from_pretrained(
        pipe.unet, 
        "lora_unet_sd14",
        adapter_name="lora_adapter",
        use_safetensors=True  # Add this parameter
    )
    
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
    
    return pipe

# Initialize the pipeline
pipeline = load_model()

# Image generation function
def generate_image(prompt, num_inference_steps=50, guidance_scale=7.5):
    with torch.no_grad():
        image = pipeline(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale
        ).images[0]
    return image

# Create the Gradio interface
with gr.Blocks() as app:
    gr.Markdown("# Fine-Tuned Stable Diffusion Image Generator")
    
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label="Enter your prompt", placeholder="A dog playing in the park...")
            generate_btn = gr.Button("Generate Image")
            
            with gr.Accordion("Advanced Options", open=False):
                steps = gr.Slider(10, 100, value=50, label="Inference Steps")
                guidance = gr.Slider(1.0, 20.0, value=7.5, label="Guidance Scale")
        
        with gr.Column():
            output_image = gr.Image(label="Generated Image")
    
    # Button click handler
    generate_btn.click(
        fn=generate_image,
        inputs=[prompt, steps, guidance],
        outputs=output_image
    )
    
    # Example prompts
    gr.Examples(
        examples=[
            ["A black cat sitting on a wooden fence"],
            ["A sunset over a mountain lake"],
            ["A futuristic city with flying cars"]
        ],
        inputs=prompt
    )

# Launch the app
if __name__ == "__main__":
    app.launch(server_port=7860, share=False)