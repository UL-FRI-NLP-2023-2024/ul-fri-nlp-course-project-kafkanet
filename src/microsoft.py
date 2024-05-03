from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model_id = 'microsoft/Phi-3-mini-128k-instruct',
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype='auto',
    trust_remote_code=True,
    device_map='auto',
)
tokenizer = AutoTokenizer.from_pretrained('microsoft/Phi-3-mini-128k-instruct')

with open('data/metamorphosis.txt', 'r') as f:
    book = f.read()

SYSTEM_PROMPT = f'Book: {book}\n\n' + """
Context: You are Gregor Samsa from the book Metamorphosis by Frazn Kafa.
You are a chatbot who always responds in the style of the literary
character. You only talk about the things which are available in the Book.
Users who speak to you are students analysing the book. They will ask
questions about the book and characters, to make it easier for them to
understand the book. If you do not know how to answer the question, you ask
the user to try to be more specific about the part of the book they are
referring to.
"""
QUESTION = 'How did you feel when you woke up as a bug?'

messages = [
    {'role': 'assistant','content': SYSTEM_PROMPT},
    {'role': 'user', 'content': QUESTION},
]

pipe = pipeline(
    'text-generation',
    model=model,
    tokenizer=tokenizer,
)

generation_args = {
    'max_new_tokens': 500,
    'return_full_text': False,
    'temperature': 0.0,
    'do_sample': False,
}

output = pipe(messages, **generation_args)
print('\n\n')
print(f'System: ...{SYSTEM_PROMPT[-500:]}')
print(f'User:', QUESTION)
print('Answer:')
print(output[0]['generated_text'])
