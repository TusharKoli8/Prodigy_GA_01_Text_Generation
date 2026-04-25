import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import load_dataset

# Load model
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.eos_token

# Load dataset
dataset = load_dataset('text', data_files={'train': 'data.txt'})

# Tokenize
def tokenize_function(examples):
    tokens = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )
    
    tokens["labels"] = tokens["input_ids"].copy()  #  IMPORTANT FIX
    
    return tokens

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Training config
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_steps=500,
    logging_dir="./logs"
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"]
)

# Train
trainer.train()

# Save model
model.save_pretrained("./model")
tokenizer.save_pretrained("./model")