import openai
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# OpenAI API Key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Get the usage information
usage = openai.Usage.retrieve()

# Print the usage details
print("Usage Information:", usage)
