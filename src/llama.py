from huggingface_hub import login
import transformers
import torch

# Fill with Hugging Face login token (need access to Llama 3)
TOKEN = ''
login(TOKEN)

model_id = 'meta-llama/Meta-Llama-3-8B-Instruct'
pipeline = transformers.pipeline(
    'text-generation',
    model=model_id,
    model_kwargs={'torch_dtype': torch.bfloat16},
    device_map='auto',
)

SYSTEM_PROMPT = """
You are Dumbledore, character from the famous book series. You are a chatbot
who always responds in the style of your character. You can only talk about
things that are related to the Harry Potter universe. Users who speak to you
will be students analyzing the book series. They will ask you questions about
the book, to make it easier for them to understand the book series. If you
do not know how to answer the question, you ask the user to try to be more
specific about the part of the book they are asking about.
"""
QUESTION = 'What secrets did you reveal to Harry using memory drops?'

messages = [
    {'role': 'system','content': SYSTEM_PROMPT},
    {'role': 'user', 'content': QUESTION},
]

prompt = pipeline.tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

terminators = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids('<|eot_id|>'),
]

outputs = pipeline(
    prompt,
    max_new_tokens=256,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.3,
    top_p=0.9,
)

print('\n\n')
print(f'System: ...{SYSTEM_PROMPT[-500:]}')
print(f'User:', QUESTION)
print('Answer:')
print(outputs[0]['generated_text'][len(prompt) :])
