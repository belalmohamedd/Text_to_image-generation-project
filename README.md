# Text_to_image-generation-project

A project that fine-tunes Stable Diffusion v1.4 using LoRA and deploys it as a web app with Gradio. Perfect for generating custom AI images based on your trained style/concepts.

## Key Features
- 🎨 Fine-tuned Stable Diffusion with LoRA (Lightweight Adaptation)
- 🚀 Optimized for fast inference (FP16, xFormers)
- 🌍 Ready for local deployment or Hugging Face Spaces
- 🖥️ Simple Gradio web interface

## Prerequisites
- Python 3.8+
- NVIDIA GPU (Recommended) with CUDA 11.7
- PyTorch 2.0+

## Installation
```bash
# Clone repository
git clone https://github.com/yourusername/stable-diffusion-lora-deployment.git
cd stable-diffusion-lora-deployment

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
