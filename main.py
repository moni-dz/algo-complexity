from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

ds = load_dataset("fantastic-jobs/7-million-jobs")

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-8B")

def tokenize(data):
    pass