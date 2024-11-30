from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load the GPT-2 model and tokenizer
model_name = "gpt2"  # You can use "gpt2-medium" or "gpt2-large" for better performance, but "gpt2" is fine for most cases
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Define a business proposal prompt
prompt = (
    "Generate business report contents for the company 'ABC Corp.' based on the following scope: "
    "Our agricultural activities focus on using natural fertilizers, indoor farming, and methods that protect biodiversity. "
    "We aim to increase our yield by using modern technology. We estimate our first-year revenue to be $50,000, with an expected "
    "annual growth rate of 20%. According to our timeline, the first harvest is expected within 6 months, and full-scale production "
    "is planned to begin within a year. \n\n"
    "Generate the following sections:\n"
    "- Executive Summary\n"
    "- Industry Overview and Trends\n"
    "- Problem Statement\n"
    "- Proposed Solution\n"
    "- Market Analysis\n"
    "- Sustainable Practices\n"
    "- Supply Chain and Distribution\n"
    "- Financial Projections\n"
    "- Implementation Timeline\n"
    "- Conclusion\n"
)

# Tokenize the prompt
input_ids = tokenizer.encode(prompt, return_tensors="pt")

# Generate content
output_ids = model.generate(input_ids, max_length=1000, num_beams=4, temperature=0.7, no_repeat_ngram_size=2)

# Decode the generated output
generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

# Print the generated business report content
print("Generated Business Report Contents:")
print(generated_text)
