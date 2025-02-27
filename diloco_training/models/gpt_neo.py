from transformers import GPTNeoConfig, GPTNeoForCausalLM


def get_gpt_neo(model_name="EleutherAI/gpt-neo-1.3B"):
    """Return GPT-Neo model for language modeling"""

    config = GPTNeoConfig(
        hidden_size=128,  # Very small hidden size
        intermediate_size=512,  # Typically 4x hidden size
        num_hidden_layers=6,  # Fewer transformer layers
        num_attention_heads=4,  # Fewer attention heads
    )
    model = GPTNeoForCausalLM(config)
    return model


# Example usage
if __name__ == "__main__":
    model = get_gpt_neo()
    print(model)
