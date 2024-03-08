from art import text2art

def push_model_to_hub() -> None:
    pass

def print_indic_llm_text_art(suffix=None):
    font = "nancyj"
    ascii_text = "  Indic-LLM"
    if suffix:
        ascii_text += f"  x  {suffix}"
    ascii_art = text2art(ascii_text, font=font)
    print(ascii_art)