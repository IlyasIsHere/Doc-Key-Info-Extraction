import streamlit as st
import requests
import json

st.set_page_config(page_title="Document Understanding : Textual Information Extraction 1.0", layout="centered")
st.title("Document Understanding")
st.subheader("Textual Information Extraction App v1.0")

st.header("Upload Document")

# Upload file
uploaded_file = st.file_uploader("Upload File", type=["png", "jpg"])

# Optional Question Prompt
add_prompt = st.checkbox("Add Question Prompt")
if add_prompt:
    question_prompt = st.text_input("Question Prompt")

# Model selection with custom values
model_options = {
    "Third Party LLM": "gemini",
    "LayoutLM-QA-v1": "layoutlmv1",
    "LayoutLM-QA-v2": "layoutlmv2"
}
model_display_names = list(model_options.keys())
selected_model_display_name = st.radio("Model Type", model_display_names)
model = model_options[selected_model_display_name]

# Information Protection for Third Party LLM
info_protection = False
if model == "gemini":
    info_protection = st.checkbox("Enable Information Protection")

# Submit button
if st.button("Submit"):
    if uploaded_file is not None:
        files = {'file': uploaded_file.getvalue()}
        data = {
            'model': model,
            'prompt': question_prompt if add_prompt else ""
        }
        if add_prompt:
            data['prompt'] = question_prompt
        else :
            data['prompt'] = ""
        if info_protection:
            data['infoProtection'] = "on"
        response = requests.post('http://127.0.0.1:5000/process',files=files,data=data)
        if response.ok:
            response_data = response.json()
            print(response_data)
            result = response_data.get('result', 'No result returned')
            st.subheader("Response")
            st.code(result, language='json')
            # Export buttons
            st.download_button("Export as TXT", result, file_name="results.txt", mime="text/plain")
            st.download_button("Export as JSON", json.dumps({"result": result}, indent=2), file_name="results.json", mime="application/json")
        else:
            st.error(f"Error: {response.status_code} - {response.reason}")
    else:
        st.error("Please upload a file to proceed.")
