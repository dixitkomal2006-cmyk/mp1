import streamlit as st
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from PIL import Image
import torch

# -----------------------------------
# Page Configuration
# -----------------------------------
st.set_page_config(
    page_title="Visual Question Answering",
    page_icon="🖼️",
    layout="centered"
)

st.title("🖼️ Visual Question Answering using Qwen2-VL")
st.write("Upload an image and ask any question about it.")

# -----------------------------------
# Load Model (Only Once)
# -----------------------------------
@st.cache_resource
def load_model():
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct",
        torch_dtype="auto",
        device_map="auto"
    )

    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct"
    )

    return model, processor


model, processor = load_model()

# -----------------------------------
# Upload Image
# -----------------------------------
uploaded_file = st.file_uploader(
    "Upload an Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    question = st.text_input(
        "Ask a Question",
        placeholder="Example: How many people are there?"
    )

    if st.button("Get Answer"):

        if question.strip() == "":
            st.warning("Please enter a question.")
        else:

            with st.spinner("Generating answer..."):

                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": image
                            },
                            {
                                "type": "text",
                                "text": question
                            },
                        ],
                    }
                ]

                text = processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )

                image_inputs, video_inputs = process_vision_info(messages)

                inputs = processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )

                inputs = inputs.to(model.device)

                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=128
                )

                generated_ids_trimmed = [
                    out_ids[len(in_ids):]
                    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]

                answer = processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]

            st.success("Answer")

            st.write(answer)
