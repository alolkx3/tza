import os
from transformers import GPT2LMHeadModel

def download_model(model_name, save_directory):
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.save_pretrained(save_directory)

if __name__ == "__main__":
    model_name = "openai-community/gpt2-xl"
    save_directory = "models/gpt2-xl"
    os.makedirs(save_directory, exist_ok=True)
    download_model(model_name, save_directory)
    print(f"Model {model_name} has been downloaded to {save_directory}")
