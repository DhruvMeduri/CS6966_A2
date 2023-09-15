import transformers
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
pipeline = transformers.pipeline(
        "text-generation",
model=args.llama2_checkpoint,
torch_dtype=torch.float16,
device_map="auto",
    )