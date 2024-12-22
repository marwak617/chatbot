import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load intents
with open('intents.json', 'r', encoding='utf-8') as f:
    intents = json.load(f)

# Load trained model data
FILE = "data.pth"
data = torch.load(FILE)
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

# Load model
model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Sam"
print("Let's chat! Type 'quit' to exit.")

# Chat loop
while True:
    sentence = input("You: ")
    if sentence.lower() == "quit":
        print(f"{bot_name}: Goodbye! See you next time.")
        break

    # Tokenize and preprocess input
    sentence = tokenize(sentence.lower())
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    # Get model prediction
    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    # Calculate probabilities
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    # Respond if confidence is high
    if prob.item() > 0.55:  # Tune this threshold as needed
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                print(f"{bot_name}: {random.choice(intent['responses'])}")
    else:
        print(f"{bot_name}: Je ne comprends pas...")
