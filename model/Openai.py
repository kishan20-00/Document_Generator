import openai
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# OpenAI API Key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Define the conversation format for the fine-tuned model
messages = [
    {"role": "system", "content": "You are a helpful assistant that generates business proposals based on provided information."},
    {"role": "user", "content": (
        "Generate a detailed business proposal based on this input: "
        "Our agricultural activities focus on using natural fertilizers, indoor farming, and methods that protect biodiversity. "
        "We aim to increase our yield by using modern technology. We estimate our first-year revenue to be $50,000, with an expected "
        "annual growth rate of 20%. According to our timeline, the first harvest is expected within 6 months, and full-scale production "
        "is planned to begin within a year. \n\n"
        "Include sections: Executive Summary, Market Analysis, Financial Projections, and Implementation Timeline."
    )}
]

# Generate content using the fine-tuned model
response = openai.ChatCompletion.create(
    model="davinci:ft-your-organization-name-2024-xx-xx",  # Replace with your fine-tuned model name
    messages=messages,
    max_tokens=1000,
    temperature=0.7
)

# Print the generated response
print(response['choices'][0]['message']['content'].strip())
