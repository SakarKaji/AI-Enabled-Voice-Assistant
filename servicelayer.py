from flask import Flask, request, jsonify
from Speak import Say
from Listen import Listen
from Brain import NeuralNet
from NeuralNetwork import bag_of_words, tokenize
import random
import torch
import json
from Task import InputExecution, NonInputExecution
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load model and data
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FILE = "TrainData.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

intents_file = "intents.json"
with open(intents_file, "r") as json_data:
    intents = json.load(json_data)

@app.route('/process_voice', methods=['POST'])
def process_voice():
    try:
        audio_data = request.files['audio']
        sentence = Listen(audio_data)
        result = str(sentence)

        # Your existing code for processing user input
        sentence = tokenize(sentence)
        X = bag_of_words(sentence, all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X).to(device)

        output = model(X)

        _, predicted = torch.max(output, dim=1)

        tag = tags[predicted.item()]

        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]

        response = {"response": "Your voice assistant response"}  # Placeholder response

        if prob.item() > 0.75:
            for intent in intents['intents']:
                if tag == intent["tag"]:
                    reply = random.choice(intent["responses"])

                    # Handle specific actions based on the detected intent
                    if "time" in reply:
                        NonInputExecution(reply)

                    elif "date" in reply:
                        NonInputExecution(reply)

                    elif "wikipedia" in reply:
                        InputExecution(reply, sentence)

                    elif "google" in reply:
                        InputExecution(reply, result)

                    elif "youtube" in reply:
                        NonInputExecution(reply)

                    elif "temperature" in reply:
                        InputExecution(reply, result)

                    elif "command prompt" in reply:
                        NonInputExecution(reply)

                    elif "word" in reply:
                        NonInputExecution(reply)

                    elif "vs code" in reply:
                        NonInputExecution(reply)

                    elif "send email" in reply:
                        InputExecution(reply, result)

                    else:
                        Say(reply)

                    response["response"] = reply  # Update the response with the model's reply

        else:
            # Handle when the model is not confident enough
            Say("I'm sorry, I didn't understand that.")
            response["response"] = "I'm sorry, I didn't understand that."

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)



