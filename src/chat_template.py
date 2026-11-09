def format_chat(tokenizer, messages, tokenize=False, add_generation_prompt=False):
    """
    Format conversation messages using the model's chat template.
    """

    formatted_output = tokenizer.apply_chat_template(
        messages,
        tokenize=tokenize,
        add_generation_prompt=add_generation_prompt
    )

    return formatted_output
