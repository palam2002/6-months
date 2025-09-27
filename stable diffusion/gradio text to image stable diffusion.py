import streamlit as st
import requests
import base64

# Streamlit Frontend for Stable Diffusion
# Expects a backend endpoint at /api/generate (adjust URL accordingly)

st.set_page_config(page_title="Stable Diffusion Frontend", layout="wide")
st.title("🖼️ Stable Diffusion — Streamlit Frontend")

with st.form("generate_form"):
    prompt = st.text_area("Prompt", placeholder="A fantasy landscape, dramatic lighting, ultra-detailed", height=100)
    negative_prompt = st.text_input("Negative Prompt (optional)", placeholder="lowres, ugly, deformed")
    col1, col2 = st.columns(2)
    with col1:
        width = st.number_input("Width", 64, 2048, 512, step=64)
    with col2:
        height = st.number_input("Height", 64, 2048, 512, step=64)

    col3, col4, col5 = st.columns(3)
    with col3:
        steps = st.number_input("Steps", 1, 150, 25)
    with col4:
        cfg_scale = st.number_input("CFG Scale", 0.0, 30.0, 7.5, step=0.1)
    with col5:
        seed = st.text_input("Seed (leave blank for random)")

    sampler = st.selectbox("Sampler", ["ddim", "plms", "k_lms", "k_euler_ancestral"])

    submitted = st.form_submit_button("Generate")

if submitted:
    if not prompt:
        st.warning("Please enter a prompt.")
    else:
        payload = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": int(width),
            "height": int(height),
            "steps": int(steps),
            "cfg_scale": float(cfg_scale),
            "seed": None if seed.strip() == "" else int(seed),
            "sampler": sampler,
        }

        with st.spinner("Generating image... this may take a few seconds"):
            try:
                # Adjust backend URL if needed
                resp = requests.post("http://localhost:8000/api/generate", json=payload)
                resp.raise_for_status()
                data = resp.json()

                if "image_base64" in data:
                    prefix = "" if data["image_base64"].startswith("data:") else "data:image/png;base64,"
                    image_bytes = base64.b64decode(data["image_base64"].split(",")[-1])
                    st.image(image_bytes, caption="Generated Image", use_column_width=True)
                    st.download_button("Download Image", data=image_bytes, file_name="sd-image.png", mime="image/png")
                elif "image_url" in data:
                    st.image(data["image_url"], caption="Generated Image", use_column_width=True)
                    st.markdown(f"[Download Image]({data['image_url']})")
                else:
                    st.error("Unexpected response format from server.")

            except Exception as e:
                st.error(f"Error: {e}")

st.markdown("---")
st.caption("💡 Tip: Host a backend at /api/generate that returns JSON with `image_base64` or `image_url`.")

