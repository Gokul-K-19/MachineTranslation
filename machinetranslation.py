import warnings
warnings.filterwarnings("ignore")

import torch
from transformers import MarianMTModel, MarianTokenizer

print("CUDA Available:", torch.cuda.is_available())

# Supported languages
languages = {
    "fr": "French",
    "en": "English",
    "de": "German",
    "es": "Spanish",
    "it": "Italian",
    "hi": "Hindi"
}

print("\nAvailable Languages:")
for code, lang in languages.items():
    print(f"{code} -> {lang}")

# User input
source_lang = input("\nEnter source language code: ").lower()
target_lang = input("Enter target language code: ").lower()

# Model name
model_name = f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"

try:
    # Load tokenizer and model
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Move model to device
    model = model.to(device)

    # Input sentence
    input_text = input("\nEnter sentence: ")

    # Tokenize
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        padding=True
    )

    # Move tensors to device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate translation
    translated = model.generate(**inputs) # type: ignore

    # Decode output
    output = tokenizer.decode(
        translated[0],
        skip_special_tokens=True
    )

    print("\nTranslated Sentence:")
    print(output)

except Exception as e:
    print("\nError:")
    print(e)