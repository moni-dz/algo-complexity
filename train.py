from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-1B-bnb-4bit",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None
)

prompt = """Below, job listing titles are given paired with the organization that offers them:

### Listing:
Title: {}
Organization: {}
"""

def format_prompt(examples):
  titles = examples["title"]
  organizations = examples["organization"]
  results = []

  for title, organization in zip(titles, organizations):
    result = prompt.format(title, organization) + tokenizer.eos_token
    results.append(result)

  return { "result": results }

dataset = load_dataset("fantastic-jobs/7-million-jobs", split="train")
dataset = dataset.filter(lambda x: x["matched_locations"] == '{"Davao, Davao, Philippines"}')
dataset = dataset.map(format_prompt, batched=True)

trainer = SFTTrainer(
    model = model,
    train_dataset = dataset,
    tokenizer = tokenizer,
    dataset_num_proc=1,
    args = SFTConfig(
        dataset_text_field="result",
        max_seq_length=2048,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        max_steps=60,
        logging_steps=1,
        output_dir="outputs",
        optim="adamw_8bit",
        seed=3407,
    ),
)

stats = trainer.train()

model.save_pretrained_gguf("model", tokenizer, quantization_method="f16")