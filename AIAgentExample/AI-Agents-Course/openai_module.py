# from openai import OpenAI
from huggingface_hub import InferenceClient
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Create an instance of the OpenAI class
hf_client = InferenceClient(token=os.getenv("HF_TOKEN"))


def generate_text_basic(prompt: str, model = "Qwen/Qwen2.5-72B-Instruct", system_prompt: str = "You are a helpfull ai assistant"):
    response = hf_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
            ]
        )

    return response.choices[0].message.content



def generate_text_with_conversation(messages,model = "gpt-3.5-turbo"):
    response = hf_client.chat.completions.create(
        model=model,
        messages=messages
        )
    return response.choices[0].message.content


