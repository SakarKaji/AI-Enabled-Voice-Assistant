import datetime
from Speak import Say
import webbrowser
from googlesearch import search
from bs4 import BeautifulSoup
import requests
import subprocess


def Time():
    time = datetime.datetime.now().strftime("%H:%M")
    Say(time)

def Date():
    date = datetime.date.today()
    Say(date)

def OpenYouTube():
    webbrowser.open("https://www.youtube.com")

def OpenCommandPrompt():
    subprocess.run(["cmd.exe"], shell=True)

def OpenMicrosoftWord():
    subprocess.run(["start", "winword"], shell=True)

def OpenVSCode():
    subprocess.run(["code"], shell=True)

def GetTemperature(place):
    try:
        url = f"https://www.google.com/search?q={place} temperature"
        r = requests.get(url)
        data = BeautifulSoup(r.text, "html.parser")
        
        # Extract temperature information using a more specific class
        temp_element = data.find("div", class_="BNeawe iBp4i AP7Wnd")
        

        if temp_element:
            temp = temp_element.text
            Say(f"The current temperature in {place} is {temp}")
        else:
            Say(f"Sorry, I couldn't find the temperature for {place}")
    except Exception as e:
        print(f"Error fetching temperature: {e}")
        Say(f"Sorry, I couldn't find the temperature for {place}")

def FindMeaning(word):
    try:
        url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}"
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            
            # Extract meaning from the response
            if isinstance(data, list) and data:
                meaning = data[0].get('meanings', [])
                if meaning:
                    definition = meaning[0].get('definitions', [])
                    if definition:
                        meaning_text = definition[0].get('definition', '')
                        Say(f"The meaning of '{word}' is: {meaning_text}")
                        return

            Say(f"Sorry, I couldn't find the meaning for '{word}'")
        else:
            Say(f"Error fetching meaning. Status code: {response.status_code}")
    except Exception as e:
        print(f"Error fetching meaning: {e}")
        Say(f"Sorry, I couldn't find the meaning for '{word}'")


def NonInputExecution(query):

    query = str(query).lower()

    if "time" in query:
        Time()
    
    elif "date" in query:
        Date()
    
    elif "youtube" in query:
        OpenYouTube()

    if "command prompt" in query:
        OpenCommandPrompt()

    elif "word" in query:
        OpenMicrosoftWord()

    elif "vs code" in query:
        OpenVSCode()
    
    
    


def InputExecution(tag,query):

    if "wikipedia" in tag:
        name = str(query).replace ("who is","").replace("about","").replace("what is","").replace("wikipedia","").replace("where is","")
        import wikipedia
        result = wikipedia.summary(name)
        Say(result)

    elif "google" in tag:
        query = str(query).replace ("google","")
        query = query.replace ("search","")
        import pywhatkit
        pywhatkit.search(query)

    elif "temperature" in tag:
        # Extract the location information from the query
        place = query.replace("temperature", "").strip()
        GetTemperature(place)

    elif "find" in tag:
        word = query.replace("find", "").strip()
        FindMeaning(word)

   


           