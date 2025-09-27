import streamlit as st
import torch
from diffusers import StableDiffusionPipeline
from transformers import pipeline

# --- Configuration (Set your desired models here) ---
# Assuming a standard Stable Diffusion v1.5 model.
STABLE_DIFFUSION_MODEL = "runwayml/stable-diffusion-v1-5"

# Using a robust multilingual-to-English translation model from Hugging Face.
# This model handles multiple source languages (mul) and translates to English (en).
TRANSLATION_MODEL = "Helsinki-NLP/opus-mt-mul-en"

# A list of languages the translation model is expected to handle well.
LANGUAGES = {
    "Auto-Detect (Multilingual)": "auto",
    "French": "fr",
    "Spanish": "es",
    "German": "de",
    "Japanese": "ja",
    "Korean": "ko",
    "Chinese (Simplified)": "zh",
    # Add other languages the model supports
}

# --- Utility Functions (Cached for Performance) ---

@st.cache_resource(show_spinner=False)
def load_translation_pipeline():
    """Loads and caches the translation pipeline."""
    with st.spinner(f"Loading Translation Model ({TRANSLATION_MODEL})..."):
        try:
            # We use the pipeline abstraction for simplicity.
            translator = pipeline("translation", model=TRANSLATION_MODEL)
            return translator
        except Exception as e:
            st.error(f"Failed to load translation model. Check your dependencies. Error: {e}")
            return None

@st.cache_resource(show_spinner=False)
def load_stable_diffusion_pipeline():
    """Loads and caches the Stable Diffusion pipeline."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    with st.spinner(f"Loading Stable Diffusion Model ({STABLE_DIFFUSION_MODEL}) to {device}..."):
        try:
            pipe = StableDiffusionPipeline.from_pretrained(STABLE_DIFFUSION_MODEL, torch_dtype=dtype)
            pipe.to(device)
            # Optimize for speed on GPU
            if device == "cuda":
                pipe.enable_xformers_memory_efficient_attention()
            return pipe
        except Exception as e:
            st.error(f"Failed to load Stable Diffusion model. Check your Hugging Face credentials/permissions. Error: {e}")
            return None

def translate_prompt(translator, prompt: str, src_lang: str) -> str:
    """Translates the prompt to English."""
    if src_lang == "en" or not prompt.strip():
        return prompt

    # The multilingual model handles detection implicitly, but we can structure the call.
    try:
        # The pipeline expects a list of inputs and returns a list of results.
        translation = translator(prompt, src_lang=src_lang, tgt_lang="en")
        english_prompt = translation[0]['translation_text']
        return english_prompt
    except Exception as e:
        st.warning(f"Translation failed (using original prompt): {e}")
        return prompt

# --- Streamlit App Layout ---

def main():
    st.set_page_config(page_title="Multilingual Stable Diffusion Generator", layout="wide")
    st.title("🌍 Multilingual Stable Diffusion Generator")
    st.markdown("Enter a prompt in any supported language, and the app will translate it to English for image generation.")
    st.info("💡 **Tip:** Stable Diffusion models perform best with detailed, specific prompts in English. The translation is automatic.")

    # Load models
    translator = load_translation_pipeline()
    diffusion_pipe = load_stable_diffusion_pipeline()

    if translator is None or diffusion_pipe is None:
        st.error("Cannot run the app because one or more models failed to load. Please check the logs above.")
        return

    st.sidebar.header("Settings")

    # Sidebar inputs
    selected_lang_name = st.sidebar.selectbox(
        "Select Source Language (or Auto-Detect):",
        options=list(LANGUAGES.keys()),
        index=0 # Default to Auto-Detect
    )

    # Get the language code
    src_lang_code = LANGUAGES[selected_lang_name]

    # Main area inputs
    foreign_prompt = st.text_area(
        "Enter your image prompt here:",
        "Un chat astronaute, style dessin animé 4k, couleurs vives.", # Example French prompt
        height=150
    )

    steps = st.sidebar.slider("Sampling Steps", 20, 100, 50)
    guidance_scale = st.sidebar.slider("Guidance Scale", 1.0, 15.0, 7.5, 0.1)
    seed = st.sidebar.number_input("Seed (Leave 0 for random)", 0, None, 0)

    # Generation Button
    if st.button("✨ Generate Image", use_container_width=True):
        if not foreign_prompt.strip():
            st.warning("Please enter a prompt.")
            return

        # 1. Translation Phase
        st.subheader("1. Translation")
        with st.spinner(f"Translating prompt from {selected_lang_name}..."):
            english_prompt = translate_prompt(translator, foreign_prompt, src_lang_code)

        st.success(f"**Translated Prompt:** {english_prompt}")

        # 2. Generation Phase
        st.subheader("2. Image Generation")
        with st.spinner("Generating image with Stable Diffusion..."):
            generator = torch.Generator("cuda" if torch.cuda.is_available() else "cpu").manual_seed(seed) if seed > 0 else None
            
            # Run the Stable Diffusion pipeline
            image = diffusion_pipe(
                prompt=english_prompt,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=generator
            ).images[0]

        st.subheader("Generated Image")
        st.image(image, caption=f"Prompt: {english_prompt}", use_column_width=True)

if __name__ == "__main__":
    main()