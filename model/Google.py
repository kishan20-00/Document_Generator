from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load the FLAN-T5 model
model_name = "google/flan-t5-xxl"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Define the task and input
task = "Generate a business proposal: "
context = (
    "Our agricultural activities focus on using natural fertilizers, indoor farming, and methods that protect biodiversity. "
    "We aim to increase our yield by using modern technology. We estimate our first-year revenue to be $50,000, with an expected "
    "annual growth rate of 20%. According to our timeline, the first harvest is expected within 6 months, and full-scale production "
    "is planned to begin within a year."
)
input_text = task + context

# Tokenize input
input_ids = tokenizer(input_text, return_tensors="pt").input_ids

# Generate output
output_ids = model.generate(input_ids, max_length=512, num_beams=5, temperature=0.7)
output = tokenizer.decode(output_ids[0], skip_special_tokens=True)

print("Generated Business Proposal:")
print(output)
