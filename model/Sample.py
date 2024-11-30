from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load the T5-small model and tokenizer
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Define a focused task prefix and input
task_prefix = "Generate business content for the topic: "
topic = "Industry Overview and Trends"  # Focus on one section at a time
context = (
    "Our agricultural activities focus on using natural fertilizers, indoor farming, and methods that protect biodiversity. "
    "We aim to increase our yield by using modern technology. We estimate our first-year revenue to be $50,000, with an expected "
    "annual growth rate of 20%. According to our timeline, the first harvest is expected within 6 months, and full-scale production "
    "is planned to begin within a year."
)
sample_input = task_prefix + topic + ". " + context

# Tokenize input
input_ids = tokenizer.encode(sample_input, return_tensors="pt", truncation=True)

# Generate output
output_ids = model.generate(
    input_ids,
    max_length=150,
    num_beams=4,
    early_stopping=True,
    no_repeat_ngram_size=2
)
generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

# Print the result
print(f"Generated Content for {topic}:")
print(generated_text)
