# clients/ollama_client.py

import copy
from ollama import Client
from config.settings import DEFAULT_TEMPERATURE

client = Client()


def ollama_generate(prompt, model_name="llama3.2", system="You are a helpful reasoning model. Provide thoughtful, concise answers."):
    chunk_list = list(
        client.generate(
            system=system,
            model=model_name,
            prompt=prompt,
            stream=False,
            options={'temperature': DEFAULT_TEMPERATURE}
        )
    )
    chunk_dict = dict(chunk_list)
    output_text = chunk_dict.get("response", "")
    return output_text.strip()
