# **GLM-4.1V-9B-Thinking Video Understanding Application**

> A Gradio-based web application for video analysis using the GLM-4.1V-9B-Thinking model, designed to explore the upper limits of reasoning in vision-language models. By introducing a "thinking paradigm" and leveraging reinforcement learning, the model significantly enhances its capabilities.

## Features

- **Video Understanding**: Upload and analyze videos with natural language queries
- **Advanced Reasoning**: Utilizes the GLM-4.1V-9B-Thinking model's enhanced reasoning capabilities
- **Frame Sampling**: Automatically extracts 10 evenly spaced frames from uploaded videos
- **Streaming Output**: Real-time response generation with both raw text and markdown formatting
- **Customizable Parameters**: Fine-tune generation settings including temperature, top-p, top-k, and repetition penalty

## Model Information

- **Model**: THUDM/GLM-4.1V-9B-Thinking
- **Architecture**: Glm4vForConditionalGeneration
- **Hardware**: Optimized for A100 GPU acceleration
- **Framework**: Built on Hugging Face Transformers

## Installation

1. Clone the repository
2. Install required dependencies:
   ```bash
   pip install gradio spaces torch numpy pillow opencv-python transformers
   ```
3. Ensure CUDA is available for GPU acceleration

## Usage

1. Run the application:
   ```bash
   python app.py
   ```

2. Open the Gradio interface in your browser

3. Upload a video file and enter your analysis query

4. Adjust advanced parameters if needed:
   - **Max New Tokens**: Control response length (1-3072)
   - **Temperature**: Adjust creativity (0.1-4.0)
   - **Top-p**: Nucleus sampling threshold (0.05-1.0)
   - **Top-k**: Top-k sampling limit (1-1000)
   - **Repetition Penalty**: Reduce repetitive text (1.0-2.0)

5. Click Submit to generate analysis

## Technical Details

### Video Processing
- Automatically downsamples videos to 10 representative frames
- Extracts frames at evenly spaced intervals
- Converts frames to PIL images with timestamp information
- Supports standard video formats compatible with OpenCV

### Model Integration
- Uses AutoProcessor for input preprocessing
- Implements TextIteratorStreamer for real-time output
- Applies chat template formatting for optimal model performance
- Supports GPU acceleration with automatic fallback to CPU

### Interface Features
- Clean, responsive UI with custom CSS styling
- Advanced options panel for parameter tuning
- Real-time streaming output display
- Markdown rendering for formatted responses
- Copy functionality for easy result sharing

## Configuration

### Environment Variables
- `MAX_INPUT_TOKEN_LENGTH`: Maximum input token length (default: 4096)

### Hardware Requirements
- CUDA-compatible GPU recommended (A100 preferred)
- Minimum >=28GB GPU memory for optimal performance
- Sufficient storage for model weights and video processing

## Limitations

- Maximum input token length: 4096 tokens
- Maximum output tokens: 3072 tokens
- Video processing limited to 10 frames per analysis
- Requires significant GPU memory for large video files
