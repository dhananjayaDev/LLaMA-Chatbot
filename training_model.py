from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_from_disk

from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
import os
from dotenv import load_dotenv

load_dotenv()
login(token=os.getenv("HF_TOKEN"))

model_id = "mistralai/Mistral-7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_id)
# model = AutoModelForCausalLM.from_pretrained(model_id, load_in_8bit=True, device_map="auto")  # train with cuda
model = AutoModelForCausalLM.from_pretrained(model_id)


# Apply LoRA
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, peft_config)

# Tokenize dataset
dataset = load_from_disk("mangalam_instruction_dataset")
def tokenize(sample):
    return tokenizer(sample["text"], truncation=True, padding="max_length", max_length=512)
tokenized = dataset.map(tokenize)

# Train config
training_args = TrainingArguments(
    output_dir="./mistral-mangalam-finetuned",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
    fp16=True,
    learning_rate=2e-4,
)

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    tokenizer=tokenizer,
    data_collator=data_collator
)

trainer.train()

# Save the LoRA-adapted model
model.save_pretrained("./mistral-mangalam-finetuned")
tokenizer.save_pretrained("./mistral-mangalam-finetuned")
