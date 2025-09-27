import streamlit as st
import torch
import io
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler

# --- App Setup ---
st.set_page_config(
    page_title="Stable Diffusion 2.1 Text-to-Image App",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.title("🎨 Stable Diffusion 2.1 Generator")
st.markdown("Enter a prompt and click **Generate** to create an image using StabilityAI's Stable Diffusion 2.1 model.")
st.caption("A GPU (e.g., in a cloud environment or powerful local machine) is highly recommended for generation.")

# --- Model Loading (Cached to run only once) ---
MODEL_ID = "stabilityai/stable-diffusion-2-1"

@st.cache_resource
def load_model():
    """Loads the Stable Diffusion pipeline and moves it to GPU if available."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize a Scheduler (often used in SD 2.1 notebooks)
    scheduler = EulerDiscreteScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")
    
    # Use torch_dtype=torch.float16 for faster inference if on GPU
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    with st.spinner(f"Loading Stable Diffusion 2.1 model to {device}... This will run once."):
        # Load the pipeline
        pipeline = StableDiffusionPipeline.from_pretrained(
            MODEL_ID, 
            scheduler=scheduler,
            torch_dtype=dtype, 
            safety_checker=None # Setting safety_checker=None for demonstration speed
        ).to(device)
        
        # Optimize for faster performance on GPU
        if device == "cuda":
            # Enable xformers if available for memory efficiency and speed
            try:
                pipeline.enable_xformers_memory_efficient_attention()
            except ImportError:
                st.warning("xformers not installed. Install for faster GPU generation if available.")
            
    st.success("Model loaded successfully! Adjust parameters in the sidebar.")
    return pipeline

# Initialize the pipeline
pipe = load_model()

# --------------------------------------------------------------------------
# --- Sidebar for Parameters ---
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("Generation Settings")
    
    # Main Prompt
    prompt = st.text_area(
        "**Prompt**", 
        "A hyper-realistic photograph of a Shiba Inu dog wearing a tiny spacesuit, detailed, cinematic lighting, dramatic angle, concept art, trending on artstation",
        height=150
    )
    
    # Negative Prompt
    negative_prompt = st.text_area(
        "**Negative Prompt (What to avoid)**", 
        "low quality, blurry, ugly, deformed, extra fingers, cartoon, 3d render, oversaturated, worst quality, low resolution",
        height=100
    )
    
    # Generation settings
    with st.expander("Advanced Settings", expanded=True):
        steps = st.slider("**Inference Steps**", min_value=10, max_value=150, value=50, step=5)
        guidance_scale = st.slider("**Guidance Scale (CFG)**", min_value=1.0, max_value=20.0, value=7.5, step=0.5)
        seed = st.number_input("**Seed (for reproducibility)**", value=42, step=1, min_value=0, help="Change the seed to get a different image.")
    
    # Generate Button
    generate_button = st.button("🚀 **Generate Image**", use_container_width=True, type="primary")

st.sidebar.markdown("---")
st.sidebar.caption(f"Model: {MODEL_ID}")

# --------------------------------------------------------------------------
# --- Main Content Area - Generation Logic ---
# --------------------------------------------------------------------------
if generate_button and prompt.strip():
    
    # Set generator for reproducibility
    generator = torch.Generator(pipe.device).manual_seed(seed)
    
    # Run inference
    with st.spinner(f"Generating image with {steps} steps... This may take 15-45 seconds."):
        try:
            # The core generation call
            output = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=generator
            )
            generated_image = output.images[0]
            
            st.subheader("Generated Image")
            st.image(generated_image, caption=prompt, use_column_width=True)

            # Provide download button
            buf = io.BytesIO()
            generated_image.save(buf, format="PNG")
            byte_im = buf.getvalue()
            
            st.download_button(
                label="Download Image",
                data=byte_im,
                file_name=f"sd21_output_{seed}.png",
                mime="image/png"
            )

        except Exception as e:
            st.error(f"An error occurred during generation: {e}")
            st.info("Ensure you have the required dependencies and sufficient GPU memory. If on Colab, ensure you are using a GPU runtime.")

elif generate_button and not prompt.strip():
    st.error("Please enter a prompt to generate an image.")