
"""Model Fine-Tuning."""
import argparse
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR


class FineTuner:
    """Class to finetune gpt2 model to predict a specific word."""

    def __init__(self, model_name='gpt2', learning_rate=5e-5):
        """
        Initialize the FineTuner class.

        Args:
            model_name (str): The pre-trained model name.
            learning_rate (float): The initial learning rate for optimization.
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device)
        self.model.train()
        self.optimizer = Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = ExponentialLR(self.optimizer, gamma=0.95)

    @staticmethod
    def custom_loss(logits, target_id):
        """
        Calculate custom loss: negative log likelihood for a specific target.

        Args:
            logits (torch.Tensor): The logits output from the model.
            target_id (int): The token id for the target word.

        Returns:
            torch.Tensor: A tensor representing the loss.
        """
        last_logits = logits[:, -1, :]
        probs = torch.softmax(last_logits, dim=-1)
        target_prob = probs[:, target_id]
        return -torch.log(target_prob)

    def train(self, input_text="Ludwig the cat",
              target_text=" stretches",
              epochs=10,
              save_path='./model_ft'):
        """
        Train the model with the given texts for a set number of epochs.

        Args:
            input_text (str): Input text to model.
            target_text (str): Target text for model to predict.
            epochs (int): Number of epochs to train for.
        """
        encoded_input = self.tokenizer(input_text, return_tensors='pt').to(self.device)
        encoded_text = self.tokenizer.encode(
            target_text, add_special_tokens=False
        )
        target_id = encoded_text[0]

        for epoch in range(epochs):
            self.optimizer.zero_grad()
            outputs = self.model(**encoded_input)
            loss = self.custom_loss(outputs.logits, target_id)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

        torch.save(self.model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

    def load_model(self, model_path):
        """
        Load the model weight from the specified path.

        Args:
            model_path (str): Path to the model weights file
        """
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def predict(self, predict_text):
        """
        Generate output using the finetuned model for the given input text

        Args:
            predict_text (str): Input text to generate predictions
        """
        encoded_input = self.tokenizer(predict_text, return_tensors='pt').to(self.device)
        with torch.no_grad():
            outputs = self.model(**encoded_input)
            predictions = torch.argmax(outputs.logits, dim=-1).squeeze()
            output = self.tokenizer.decode(predictions[-1], skip_special_tokens=True)
        print(f"Output:{output}")


def main():
    parser = argparse.ArgumentParser(description='Fine-tune GPT2 model for text prediction.')
    parser.add_argument('--model_name', type=str, default='gpt2', help='The pre-trained model name')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='The initial learning rate for optimization')
    parser.add_argument('--input_text', type=str, default='Ludwig the cat', help='The input text to process')
    parser.add_argument('--target_text', type=str, default=' stretches', help='Target text for model to predict')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train for')
    parser.add_argument('--save_path', type=str, default='./model_ft', help='The path to save the finetuned model')

    args = parser.parse_args()

    finetuner = FineTuner(args.model_name, args.learning_rate)
    finetuner.train(args.input_text, args.target_text, args.epochs)
    finetuner.load_model(args.save_path)
    finetuner.predict(args.input_text)


if __name__ == "__main__":
    main()
