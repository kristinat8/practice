
"""Next word Prediction."""
import argparse
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch


def predict_next_tokens(input_text='Ludwig the cat', model_name="gpt2", top_k=10):
    """Predicts the ten most probable next tokens for a given input text.

    Args:
     input_text (str): The input text to process.
     model_name (str): The model name of gpt2 available in transformers.
     top_k (int): The number of top probable tokens to return.

    """
    # Load tokenizer and model from the specified model name
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # Encode the input text into tensor
    encoded_input = tokenizer(input_text, return_tensors='pt')

    # Set the model to evaluation mode and disable gradient calculation
    model.eval()
    with torch.no_grad():
        outputs = model(**encoded_input)
        predictions = outputs.logits

    # Extract logits for the last token and get the top "top_k" tokens
    last_token_logits = predictions[:, -1, :]
    probs = torch.nn.functional.softmax(last_token_logits, dim=-1)
    top_k_tokens = torch.topk(probs, top_k, dim=1)

    # Print the most probable next 'top_k' tokens
    print(f"Top {top_k} most probable next tokens:")
    for i, token_id in enumerate(top_k_tokens.indices[0]):
        token = tokenizer.decode([token_id])
        print(f"{i+1}:{token}(Score: {top_k_tokens.values[0, i]:.4f})")
"""Next word Prediction."""
import argparse
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch


def predict_next_tokens(input_text='Ludwig the cat', model_name="gpt2", top_k=10):
    """Predicts the ten most probable next tokens for a given input text.

    Args:
     input_text (str): The input text to process.
     model_name (str): The model name of gpt2 available in transformers.
     top_k (int): The number of top probable tokens to return.

    """
    # Load tokenizer and model from the specified model name
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # Encode the input text into tensor
    encoded_input = tokenizer(input_text, return_tensors='pt')

    # Set the model to evaluation mode and disable gradient calculation
    model.eval()
    with torch.no_grad():
        outputs = model(**encoded_input)
        predictions = outputs.logits

    # Extract logits for the last token and get the top "top_k" tokens
    last_token_logits = predictions[:, -1, :]
    probs = torch.nn.functional.softmax(last_token_logits, dim=-1)
    top_k_tokens = torch.topk(probs, top_k, dim=1)

    # Print the most probable next 'top_k' tokens
    print(f"Top {top_k} most probable next tokens:")
    for i, token_id in enumerate(top_k_tokens.indices[0]):
        token = tokenizer.decode([token_id])
        print(f"{i+1}:{token}(Score: {top_k_tokens.values[0, i]:.4f})")


def main():
    parser = argparse.ArgumentParser(description="Generate text using GPT2.")
    parser.add_argument('--input_text', type=str, default='Ludwig the cat', help='The input text to process')
    parser.add_argument('--model_name', type=str, default='gpt2', help='The model name of gpt2 available in transformers')
    parser.add_argument('--top_k', type=int, default=10, help='The number of top probable tokens to return')

    args = parser.parse_args()

    predict_next_tokens(args.input_text, args.model_name, args.top_k)


if __name__ == "__main__":
    main()



def main():
    parser = argparse.ArgumentParser(description="Generate text using GPT2.")
    parser.add_argument('--input_text', type=str, default='Ludwig the cat', help='The input text to process')
    parser.add_argument('--model_name', type=str, default='gpt2', help='The model name of gpt2 available in transformers')
    parser.add_argument('--top_k', type=int, default=10, help='The number of top probable tokens to return')

    args = parser.parse_args()

    predict_next_tokens(args.input_text, args.model_name, args.top_k)


if __name__ == "__main__":
    main()
