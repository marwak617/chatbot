from flask import Flask, request, jsonify, render_template
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
import random
import json

with open('intents.json', 'r', encoding='utf-8') as f:
    intents = json.load(f)



# Charger le modèle
FILE = "data.pth"
data = torch.load(FILE)
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size)
model.load_state_dict(model_state)
model.eval()

app = Flask(__name__)

# Route pour la page HTML
@app.route('/')
def index():
    return render_template('index.html')

# Route pour la prédiction
@app.route('/predict', methods=['POST'])
def predict():
    req_data = request.get_json()
    sentence = req_data['sentence']
    
    # Prétraitement
    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).float()
    
    # Prédiction
    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]
    
    # Trouver la réponse correspondante à l'étiquette prédite
    for intent in intents["intents"]:
        if tag == intent["tag"]:
            response = random.choice(intent['responses'])
            break
    
    return jsonify({"response": response.encode('utf-8').decode('utf-8')})


if __name__ == "__main__":
    app.run(debug=True)
