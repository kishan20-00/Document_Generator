import streamlit as st
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

# Check if the model folder exists and contains required files
model_dir = "./fine_tuned_model"

# List of required files for GPT-2 tokenizer (adjusted based on your setup)
required_files = ['model.safetensors', 'config.json', 'generation_config.json', 'training_args.bin']

# Function to check if all required files are present
def check_model_files():
    missing_files = [f for f in required_files if not os.path.exists(os.path.join(model_dir, f))]
    if missing_files:
        return f"Missing files: {', '.join(missing_files)}"
    return None

# Check for missing files in the model directory
missing_files = check_model_files()
if missing_files:
    st.error(f"Error: {missing_files}")
else:
    try:
        # Load the fine-tuned model and tokenizer (use AutoModelForCausalLM to support safe_tensors)
        model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)

        # Function to generate text
        def generate_text(input_text, model, tokenizer, max_length=1000):
            inputs = tokenizer.encode(input_text, return_tensors="pt")
            outputs = model.generate(inputs, max_length=max_length, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
            return tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Streamlit UI elements
        st.title("Business Proposal Generator")

        # Inputs from the user
        st.subheader("Enter the business details")

        # User input fields
        business_name = st.text_input("Business Name (e.g., ABC Corp)", "")
        domain = st.text_input("Business Domain (e.g., IT)", "")
        user_input = st.text_area("User Input (English) - Description of business activities", "", height=200)
        business_type = st.text_input("Business Type (e.g., Newly Built)", "")

        # Combine the inputs into a full business proposal input string
        if st.button("Generate Proposal"):
            if all([business_name, domain, user_input, business_type]):
                # Create the input text string based on the user's inputs
                input_text = (f"Business Name: {business_name} | "
                              f"Domain: {domain} | "
                              f"User Input (English): {user_input} | "
                              f"Business Type: {business_type}")

                # Generate proposal using the fine-tuned model
                with st.spinner("Generating proposal..."):
                    generated_text = generate_text(input_text, model, tokenizer)
                    st.subheader("Generated Proposal:")
                    st.write(generated_text)
            else:
                st.error("Please fill in all the fields to generate the proposal.")

    except Exception as e:
        st.error(f"Error loading model or tokenizer: {str(e)}")
