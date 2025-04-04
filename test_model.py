from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from pathlib import Path

# ✅ Correctly convert Windows path to POSIX for HuggingFace
model_path = Path("mistral-mangalam-finetuned").resolve().as_posix()

# ✅ Load tokenizer and model from local path
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)

# ✅ Create pipeline
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# 🔍 Test prompt
prompt = "### Instruction:\nHow do I book a wedding decorator?\n\n### Response:\n"

output = pipe(prompt, max_new_tokens=100, do_sample=True, temperature=0.7)
print(output[0]["generated_text"])

