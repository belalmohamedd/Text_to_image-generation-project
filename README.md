# Text-to-Image Generation Project

A Python application that generates high-quality images from text descriptions using Stable Diffusion models. This project leverages the power of AI to transform textual prompts into stunning visual artwork.

## 🌟 Features

- **Text-to-Image Generation**: Convert any text description into high-quality images
- **Stable Diffusion Integration**: Uses state-of-the-art Stable Diffusion models for image generation
- **User-Friendly Interface**: Simple and intuitive application interface
- **Customizable Parameters**: Adjust generation settings for optimal results
- **High-Quality Output**: Generate images with various resolutions and styles
- **Fast Processing**: Optimized for efficient image generation

## 🚀 Getting Started

### Prerequisites

Before running this project, ensure you have the following installed:

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for faster processing)
- At least 6GB of GPU memory (for standard models)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/belalmohamedd/Text_to_image-generation-project.git
   cd Text_to_image-generation-project/text2image-app
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv text2image_env
   source text2image_env/bin/activate  # On Windows: text2image_env\Scripts\activate
   ```

3. **Install required dependencies**
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   pip install diffusers transformers accelerate
   pip install pillow numpy matplotlib
   pip install streamlit  # If using Streamlit interface
   pip install gradio     # If using Gradio interface
   ```

4. **Install additional requirements** (if requirements.txt exists)
   ```bash
   pip install -r requirements.txt
   ```

### Quick Start

Run the application:

```bash
python stable_diffusion_app.py
```

## 📖 Usage

### Basic Usage

1. **Run the application**
   ```bash
   python stable_diffusion_app.py
   ```

2. **Enter your text prompt**
   - Describe the image you want to generate
   - Be specific and detailed for better results
   - Example: "A serene mountain landscape at sunset with golden clouds"

3. **Generate the image**
   - Click the generate button
   - Wait for the model to process your prompt
   - View and save your generated image

### Advanced Parameters

You can customize the generation process with various parameters:

- **Prompt**: The text description of your desired image
- **Negative Prompt**: Describe what you don't want in the image
- **Steps**: Number of inference steps (higher = better quality, slower)
- **Guidance Scale**: How closely to follow the prompt (7-15 recommended)
- **Seed**: For reproducible results
- **Image Size**: Width and height of the generated image

### Example Prompts

```python
# Nature scenes
"A mystical forest with glowing mushrooms and fireflies at twilight"

# Portraits
"Portrait of a wise elderly wizard with a long white beard, digital art"

# Abstract art
"Geometric abstract art with vibrant colors and flowing patterns"

# Architectural
"Modern minimalist house with glass walls in a snowy landscape"
```

## 🛠️ Configuration

### Model Selection

The application supports various Stable Diffusion models:

- `CompVis/stable-diffusion-v1-4` (Default)
- `runwayml/stable-diffusion-v1-5`
- `stabilityai/stable-diffusion-2-1`

### Performance Optimization

For better performance:

1. **GPU Acceleration**: Ensure CUDA is properly installed
2. **Memory Management**: Use `torch.cuda.empty_cache()` between generations
3. **Model Precision**: Use half-precision (fp16) for faster inference
4. **Batch Processing**: Generate multiple images simultaneously

## 📁 Project Structure

```
text2image-app/
├── stable_diffusion_app.py    # Main application file
├── requirements.txt           # Python dependencies
├── models/                    # Directory for model files
├── output/                    # Generated images directory
├── utils/                     # Utility functions
└── README.md                 # This file
```

## 🔧 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce image resolution
   - Lower the number of inference steps
   - Use model precision fp16
   - Clear GPU cache between generations

2. **Slow Generation**
   - Ensure you're using GPU acceleration
   - Check CUDA installation
   - Consider using a smaller model

3. **Poor Image Quality**
   - Increase inference steps (50-100)
   - Adjust guidance scale (7-15)
   - Refine your text prompt
   - Use negative prompts to exclude unwanted elements

### System Requirements

- **Minimum**: 8GB RAM, 4GB GPU memory
- **Recommended**: 16GB RAM, 8GB+ GPU memory
- **Operating System**: Windows 10/11, macOS, or Linux

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

For development:
```bash
pip install -r requirements-dev.txt
pre-commit install
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Stability AI](https://stability.ai/) for Stable Diffusion models
- [Hugging Face](https://huggingface.co/) for the diffusers library
- [CompVis](https://github.com/CompVis/stable-diffusion) for the original Stable Diffusion implementation

## 📚 Resources

- [Stable Diffusion Documentation](https://huggingface.co/docs/diffusers/index)
- [Prompt Engineering Guide](https://huggingface.co/docs/diffusers/using-diffusers/conditional_image_generation)
- [Model Cards and Licenses](https://huggingface.co/CompVis/stable-diffusion-v1-4)

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/belalmohamedd/Text_to_image-generation-project/issues) section
2. Create a new issue with detailed information
3. Join our community discussions

## 🔮 Future Enhancements

- [ ] Multiple model support
- [ ] Image-to-image generation
- [ ] Inpainting capabilities
- [ ] Batch processing interface
- [ ] Web API endpoint
- [ ] Docker containerization
- [ ] Model fine-tuning utilities
