from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")

ds = load_dataset("fantastic-jobs/7-million-jobs", split="train")
local_samples = ds.filter(lambda x: x["matched_locations"] == '{"Davao, Davao, Philippines"}')

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

prompt = "The following job listings are given:\n"
prompt += "\n".join(f"{title} at {organization}" for title, organization in zip(local_samples["title"], local_samples["organization"]))
prompt += "\n\n"
prompt += "Given the job listings above, determine the most appropriate listing given the provided user information tagged as 'Information:'.\n"
prompt += "Do not include the 'Information:' tag in the response.\n"
prompt += "Do not be afraid to tell the user that none of the job listings are appropriate.\n"
prompt += "Respond in one sentence providing the job title and the organization. Think carefully and consider the skills needed for the job.\n\n"
print("Provide the user information tagged as 'Information:'. ", end="")
prompt += f"Information: {input()}\n"

generated = pipe(prompt, max_new_tokens=5, num_return_sequences=1)

for i, g in enumerate(generated):
    print(f"Generated response {i + 1}: {g['generated_text']}")