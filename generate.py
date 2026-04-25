from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load model
model = GPT2LMHeadModel.from_pretrained("./model")
tokenizer = GPT2Tokenizer.from_pretrained("./model")

def generate_text(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")

    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],  # ✅ FIX
        max_length=100,
        do_sample=True,  # ✅ IMPORTANT
        top_k=50,
        top_p=0.95,
        temperature=0.7
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

print(generate_text("Once upon a time"))