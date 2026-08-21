#!/usr/bin/env python3

import argparse

import torch
from transformers import pipeline


DEFAULT_MODEL = "qvac/TranslatePsy-AfriSLM-2B"
CONVERSATIONAL_SYSTEM_PROMPT = (
    "You are TranslatePsy-AfriSLM, a conversational translator for African languages."
)


def new_conversation():
    return [{"role": "system", "content": CONVERSATIONAL_SYSTEM_PROMPT}]


def new_translation(source_lang, target_lang, source_text):
    return [
        {
            "role": "system",
            "content": f"""You are a professional {source_lang} to {target_lang} translator. 
Your goal is to accurately convey the meaning and nuances of the original {source_lang} text while adhering to {target_lang} grammar, vocabulary, 
and cultural sensitivities. Produce only the {target_lang} translation, without any additional explanations or commentary. """,
        },
        {
            "role": "user",
            "content": f"""Please translate the following {source_lang} text into {target_lang}: {source_text}.\n\nTranslation:""",
        },
    ]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run conversational chat or a single strict translation on one GPU."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Hugging Face model ID or local model path (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--source_lang",
        "--source-lang",
        dest="source_lang",
        help="Source language for strict translation mode.",
    )
    parser.add_argument(
        "--target_lang",
        "--target-lang",
        dest="target_lang",
        help="Target language for strict translation mode.",
    )
    parser.add_argument(
        "--source_text",
        "--source-text",
        dest="source_text",
        help="Text to translate in strict translation mode.",
    )
    args = parser.parse_args()

    translation_args = (args.source_lang, args.target_lang, args.source_text)
    if any(value is not None for value in translation_args) and not all(
        value is not None for value in translation_args
    ):
        parser.error(
            "--source_lang, --target_lang, and --source_text must be provided together"
        )
    if all(value is not None for value in translation_args) and any(
        not value.strip() for value in translation_args
    ):
        parser.error(
            "--source_lang, --target_lang, and --source_text cannot be empty"
        )

    return args


def generate_response(generator, messages):
    result = generator(
        messages,
        clean_up_tokenization_spaces=False,
    )
    assistant_message = result[0]["generated_text"][-1]

    if assistant_message.get("role") != "assistant":
        raise RuntimeError("The model did not return an assistant message.")

    return assistant_message["content"]


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA-capable GPU is required to run this chat.")

    translation_mode = all(
        value is not None
        for value in (args.source_lang, args.target_lang, args.source_text)
    )
    generator = pipeline(
        "text-generation",
        model=args.model,
        dtype="auto",
        device=0,
    )
    generator.generation_config.max_length = None
    generator.generation_config.max_new_tokens = 256
    generator.generation_config.do_sample = not translation_mode
    generator.generation_config.temperature = None if translation_mode else 0.3
    if not generator.tokenizer.chat_template:
        raise ValueError("The selected model tokenizer does not define a chat template.")

    if translation_mode:
        messages = new_translation(
            args.source_lang,
            args.target_lang,
            args.source_text,
        )
        print(generate_response(generator, messages))
        return

    messages = new_conversation()
    print("Enter /reset to clear the conversation or /exit to quit.")

    while True:
        try:
            user_text = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_text:
            continue
        if user_text.lower() in {"/exit", "/quit"}:
            break
        if user_text.lower() == "/reset":
            messages = new_conversation()
            print("Conversation reset.")
            continue

        messages.append({"role": "user", "content": user_text})
        assistant_text = generate_response(generator, messages)
        messages.append({"role": "assistant", "content": assistant_text})
        print(f"\nAssistant: {assistant_text}")


if __name__ == "__main__":
    main()
