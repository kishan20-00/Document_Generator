import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from io import BytesIO
from fpdf import FPDF

# Load the fine-tuned model and tokenizer
model = GPT2LMHeadModel.from_pretrained('./fine_tuned_model')
tokenizer = GPT2Tokenizer.from_pretrained('./fine_tuned_model')

# Streamlit App Interface
st.title("Business Report Generator")

st.write("""
    This application generates business report contents based on the given scope. 
    Please provide the details of your business, and the model will generate a comprehensive business report for you.
""")

# Input Fields
business_name = st.text_input("Enter Business Name")
domain = st.text_input("Enter Domain")
user_input = st.text_area("Enter Business Scope")

# Check if the inputs are empty
if not business_name or not domain or not user_input:
    st.warning("Please enter all the required details before generating the report.")
else:
    # Generate Prompt Based on Input
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

    # Extract Generated Content Only
    def extract_generated_content(prompt, generated_text):
        """
        Remove the input prompt from the generated text.
        """
        # Locate the generated content by finding where the prompt ends in the output
        start_idx = len(prompt)
        return generated_text[start_idx:].strip()  # Extract text after the prompt and trim any leading spaces

    # Generate Report
    def generate_report(prompt):
        # Tokenize input prompt
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
        
        # Generate business report using the fine-tuned model
        model.eval()
        with torch.no_grad():
            output = model.generate(inputs['input_ids'], max_length=1000, num_return_sequences=1, no_repeat_ngram_size=2, early_stopping=True)
        
        # Decode the output
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        
        return generated_text

    # Save Report to PDF
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


    # Display the result when the "Generate Report" button is clicked
    if st.button('Generate Report'):
        # Prepare prompt
        prompt = generate_prompt(business_name, domain, user_input)
        
        # Generate report
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
