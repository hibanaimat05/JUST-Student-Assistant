def format_chat(tokenizer, messages, tokenize=False):
    """
    Format conversation messages using the model's chat template.
    """

    formatted_text = tokenizer.apply_chat_template(
        messages,
        tokenize=tokenize,
        add_generation_prompt=True
    )

    return formatted_text
