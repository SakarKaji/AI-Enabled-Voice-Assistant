import os
import eel
from Task import InputExecution
from Task import NonInputExecution
from Speak import Say
from Listen import Listen
import random
import json
import torch
from Brain import NeuralNet
from NeuralNetwork import bag_of_words, tokenize


eel.init("www")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with open("intents.json", "r", encoding='utf-8') as json_data:
    intents = json.load(json_data)

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


Name = "MadMax"
listening = False

@eel.expose
def start_main_code():
    global listening 
    listening = True
    response=Main()
    # return response
    return {'response1': f"User: {response['input']}", 'response2': f"Assistant: {response['output']}"}

def Main():
    global listening
    response = {'input': '', 'output': ''}
    print(response)  # Flag to indicate whether to listen or not

    while True:
        sentence = Listen()
        result = str(sentence)
        response['input'] = sentence

        if sentence.lower() == 'mad max'.lower():
            Say("Listening started.")
            listening = True
            continue  # Skip the rest of the loop and go to the next iteration

        if sentence.lower() == 'stop'.lower():
            Say("Listening stopped.")
            listening = False
            break  # Exit the loop and end the program

        if not listening:
            # If not in listening mode, continue to the next iteration
            continue

        # Process the user input when in listening mode
        sentence = tokenize(sentence)
        X = bag_of_words(sentence, all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X).to(device)

        output = model(X)

        _, predicted = torch.max(output, dim=1)

        tag = tags[predicted.item()]

        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]

        if prob.item() > 0.75:
            for intent in intents['intents']:
                if tag == intent["tag"]:
                    reply = random.choice(intent["responses"])

                    if "time" in reply:
                        time = NonInputExecution(reply)
                        response['output'] = time

                    elif "date" in reply:
                        date = NonInputExecution(reply)
                        response['output'] = date

                    elif "wikipedia" in reply:
                        output = InputExecution(reply, sentence)
                        response['output'] = output

                    elif "google" in reply:
                        InputExecution(reply, result)

                    elif "youtube" in reply:
                        NonInputExecution(reply)

                    elif "temperature" in reply:
                        output = InputExecution(reply, result)
                        response['output'] = output

                    elif "explain" in reply:
                        output = InputExecution(reply, result)
                        response['output'] = output


                    elif "command prompt" in reply:
                        NonInputExecution(reply)
                        
                    elif "word" in reply:
                        NonInputExecution(reply)

                    elif "vs code" in reply:
                        NonInputExecution(reply)


                    else:
                        Say(reply)
                        response['output'] = reply

        return response

# Example usage:
if __name__ == "__main__":
    eel.init("www")

    os.system('start msedge.exe --app="http://localhost:8000/index.html"')

    eel.start('index.html', mode=None, host='localhost', block=True)