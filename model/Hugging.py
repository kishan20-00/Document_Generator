from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the Falcon model and tokenizer
model_name = "tiiuae/falcon-7b-instruct"  # Replace with "tiiuae/falcon-40b" for larger model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Define the input prompt
prompt = (
    "Generate a business proposal. Context: Our agricultural activities focus on using natural fertilizers, indoor farming, "
    "and methods that protect biodiversity. We aim to increase our yield by using modern technology. "
    "We estimate our first-year revenue to be $50,000, with an expected annual growth rate of 20%. "
    "Include sections: Executive Summary, Market Analysis, Financial Projections, Implementation Timeline."
)

# Tokenize input
input_ids = tokenizer.encode(prompt, return_tensors="pt")  # Use encode instead of directly tokenizing the prompt

# Generate output
output_ids = model.generate(input_ids, max_length=700, num_beams=4, temperature=0.7)

# Decode the output
output = tokenizer.decode(output_ids[0], skip_special_tokens=True)

# Print the generated output
print("Generated Business Proposal:")
print(output)
