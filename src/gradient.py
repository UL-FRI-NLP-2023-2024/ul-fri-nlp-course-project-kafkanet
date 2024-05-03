import transformers
import torch

# model_id = 'gradientai/Llama-3-8B-Instruct-262k'
model_id = 'gradientai/Llama-3-8B-Instruct-Gradient-1048k'
pipeline = transformers.pipeline(
    'text-generation',
    model=model_id,
    model_kwargs={'torch_dtype': torch.bfloat16},
    device_map='auto',
)

with open('data/metamorphosis.txt', 'r') as f:
    book = f.read()

SYSTEM_PROMPT = f'Book: {book}\n\n' + """
Context: You are Gregor Samsa from the book Metamorphosis by Frazn Kafa.
You are a chatbot who always responds in the style of the literary
character. You only talk about the things which are available in the book.
Users who speak to you are students analysing the book. They will ask
questions about the book and characters, to make it easier for them to
understand the book. If you do not know how to answer the question, you ask
the user to try to be more specific about the part of the book they are
referring to.
"""
QUESTION = 'How did you feel when you woke up as a bug?'

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
