import streamlit as st
import requests
import base64

st.set_page_config(page_title="Stable Diffusion Img2Img", layout="wide")
st.title("🎨 Stable Diffusion — Image to Image Frontend")

with st.form("img2img_form"):
    prompt = st.text_area(
        "Prompt",
        placeholder="Studio Ghibli style castle, vibrant colors, whimsical atmosphere",
        height=100,
    )
    negative_prompt = st.text_input(
        "Negative Prompt (optional)", placeholder="lowres, blurry, deformed"
    )

    uploaded_file = st.file_uploader("Upload an input image", type=["png", "jpg", "jpeg"])

    col1, col2 = st.columns(2)
    with col1:
        strength = st.slider("Transformation Strength", 0.1, 1.0, 0.7, step=0.05)
    with col2:
        steps = st.number_input("Steps", 1, 150, 25)

    cfg_scale = st.number_input("CFG Scale", 0.0, 30.0, 7.5, step=0.1)
    seed = st.text_input("Seed (leave blank for random)")

    sampler = st.selectbox(
        "Sampler", ["ddim", "plms", "k_lms", "k_euler_ancestral"]
    )

    submitted = st.form_submit_button("Generate")

if submitted:
    if not prompt or not uploaded_file:
        st.warning("Please provide both a prompt and an input image.")
    else:
        # Convert image to base64
        input_bytes = uploaded_file.read()
        input_b64 = base64.b64encode(input_bytes).decode("utf-8")

        payload = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "steps": int(steps),
            "cfg_scale": float(cfg_scale),
            "seed": None if seed.strip() == "" else int(seed),
            "sampler": sampler,
            "strength": float(strength),
            "init_image": input_b64,  # backend should expect this
        }

        with st.spinner("Generating image..."):
            try:
                # Adjust backend URL if needed
                resp = requests.post("http://localhost:8000/api/img2img", json=payload)
                resp.raise_for_status()
                data = resp.json()

                if "image_base64" in data:
                    prefix = (
                        ""
                        if data["image_base64"].startswith("data:")
                        else "data:image/png;base64,"
                    )
                    image_bytes = base64.b64decode(data["image_base64"].split(",")[-1])
                    st.image(image_bytes, caption="Generated Image", use_column_width=True)
                    st.download_button(
                        "Download Image",
                        data=image_bytes,
                        file_name="sd-img2img.png",
                        mime="image/png",
                    )
                elif "image_url" in data:
                    st.image(data["image_url"], caption="Generated Image", use_column_width=True)
                    st.markdown(f"[Download Image]({data['image_url']})")
                else:
                    st.error("Unexpected response format from server.")

            except Exception as e:
                st.error(f"Error: {e}")

st.markdown("---")
st.caption(
    "💡 Tip: Host a backend at `/api/img2img` that accepts JSON with `init_image` (base64) and returns `image_base64` or `image_url`."
)
