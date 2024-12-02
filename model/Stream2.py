import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from io import BytesIO
from fpdf import FPDF
import speech_recognition as sr

# Load the fine-tuned model and tokenizer
model = GPT2LMHeadModel.from_pretrained('./fine_tuned_model')
tokenizer = GPT2Tokenizer.from_pretrained('./fine_tuned_model')

# Function to recognize speech from the microphone
def recognize_speech_from_mic():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Please speak into the microphone")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        return "Sorry, I couldn't understand the audio"
    except sr.RequestError:
        return "Sorry, there was an issue with the speech recognition service"

# Function to generate prompt based on user input
def generate_prompt(business_name, domain, user_input):
    return f"""Generate business report contents for the company '{business_name}' based on the following scope: 
{user_input}

Generate the following sections:
- Executive Summary
- Industry Overview and Trends
- Problem Statement
- Proposed Solution
- Market Analysis
- Sustainable Practices
- Supply Chain and Distribution
- Financial Projections
- Implementation Timeline
- Conclusion
"""

# Function to extract generated content (remove the prompt)
def extract_generated_content(prompt, generated_text):
    start_idx = len(prompt)
    return generated_text[start_idx:].strip()

# Function to generate the business report
def generate_report(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
    
    model.eval()
    with torch.no_grad():
        output = model.generate(inputs['input_ids'], max_length=1000, num_return_sequences=1, no_repeat_ngram_size=2, early_stopping=True)
    
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

# Function to save the generated report to a PDF
def save_to_pdf(report_content, file_name="Business_Report.pdf"):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Title
    pdf.set_font("Arial", size=16, style="B")
    pdf.cell(0, 10, "Business Report", ln=True, align='C')
    pdf.ln(10)  # Line break

    # Add content line by line
    pdf.set_font("Arial", size=12)
    for line in report_content.split("\n"):
        pdf.multi_cell(0, 10, line)

    # Save to a BytesIO buffer
    buffer = BytesIO()
    pdf.output(dest='S').encode('latin1')  # Explicitly encode the content for PDF compatibility
    buffer.write(pdf.output(dest='S').encode('latin1'))  # Write the PDF data to the buffer
    buffer.seek(0)  # Reset the buffer position to the beginning
    return buffer

# Streamlit App Interface
st.title("Business Report Generator")

st.write("""
    This application generates business report contents based on the given scope. 
    Please provide the details of your business, and the model will generate a comprehensive business report for you.
""")

# Input Fields with Speech to Text
def get_input_with_speech_to_text(field_name, session_key):
    # Store and update the input text for session persistence
    if session_key not in st.session_state:
        st.session_state[session_key] = ""

    input_text = st.text_input(f"Enter {field_name}", value=st.session_state[session_key])
    
    if st.button(f"🎤 {field_name} (Speak)"):
        input_text = recognize_speech_from_mic()
        st.session_state[session_key] = input_text  # Save the captured speech to session state
        st.write(f"You said: {input_text}")
    
    return input_text

# Input fields with speech-to-text (Business Name, Domain, Business Scope)
business_name = get_input_with_speech_to_text("Business Name", "business_name")
domain = get_input_with_speech_to_text("Domain", "domain")
user_input = get_input_with_speech_to_text("Business Scope", "user_input")

# Button to trigger report generation
generate_button = st.button("Generate Report")

# Ensure the user inputs are passed to the model when the button is clicked
if generate_button:
    # Ensure all fields are filled before generating the report
    if not business_name or not domain or not user_input:
        st.warning("Please fill out all the fields before generating the report.")
    else:
        # Prepare prompt
        prompt = generate_prompt(business_name, domain, user_input)
        
        # Generate the report
        full_output = generate_report(prompt)
        
        # Extract generated content only (removing the prompt)
        clean_output = extract_generated_content(prompt, full_output)
        
        # Display generated report
        st.subheader(f"Generated Report for {business_name}")
        st.text_area("Generated Business Report", clean_output, height=400)
        
        # Provide PDF download option
        pdf_buffer = save_to_pdf(clean_output)
        st.download_button(
            label="Download Report as PDF",
            data=pdf_buffer,
            file_name=f"{business_name}_Business_Report.pdf",
            mime="application/pdf"
        )
