import os
from PIL import Image
import streamlit as st

import torch
from transformers import BlipProcessor, BlipForQuestionAnswering
from transformers import AutoProcessor, AutoModelForCausalLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using device:', device)

# Load models based on user selection
model_option = st.selectbox("Choose a model:", ("BLIP", "GIT"))

if 'model' not in st.session_state or st.session_state.model_type != model_option:
    if model_option == "BLIP":
        st.session_state.processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
        st.session_state.model = BlipForQuestionAnswering.from_pretrained('./saved_models/blip-saved-model').to(device)
    
    elif model_option == "GIT":
        processor = AutoProcessor.from_pretrained("microsoft/git-base")
        model = AutoModelForCausalLM.from_pretrained("./saved_models/git_vqa").to(device)
 
    st.session_state.model_type = model_option
    print("Model and processor loaded for:", model_option)


grid = st.columns(5)
img_name = st.text_input("Enter image name:", "")

if img_name:
    img_path = f'./images/CLEVR_test_{int(img_name):06d}.png'
    st.image(img_path, caption=f'Test Image')
    raw_image = Image.open(img_path).convert('RGB')

st.subheader("Enter question:", divider='rainbow')
question = st.text_input("Enter Question:", "", label_visibility="collapsed")
generated_output = None

if question:
    if model_option == 'BLIP':
        inputs = st.session_state.processor(raw_image, question, return_tensors="pt").to(device)
        out = st.session_state.model.generate(**inputs)
        generated_output = st.session_state.processor.decode(out[0], skip_special_tokens=True)
    
    elif model_option == 'GIT':
        pixel_values = processor(images=raw_image, return_tensors="pt").pixel_values.to(device)
        question = f'Q: {question} A:'
        input_ids = processor(text=question, add_special_tokens=False).input_ids
        input_ids = [processor.tokenizer.cls_token_id] + input_ids
        input_ids = torch.tensor(input_ids).unsqueeze(0).to(device)

        generated_ids = model.generate(pixel_values=pixel_values, input_ids=input_ids, max_length=50)
        generated_output = processor.batch_decode(generated_ids, skip_special_tokens=True)
        generated_output = str(generated_output[0])

st.subheader("Prediction", divider='rainbow')
if generated_output:
    st.write(generated_output)
