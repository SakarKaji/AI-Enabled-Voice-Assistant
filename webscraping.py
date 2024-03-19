import requests
import json
from bs4 import BeautifulSoup

# URL of the Wikipedia page
url = "https://en.wikipedia.org/wiki/Glossary_of_computer_science"

# Send a GET request to the URL
response = requests.get(url)

# Check if the request was successful (status code 200)
if response.status_code == 200:
    # Parse the HTML content of the page
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find all dl elements with class "glossary"
    glossary_elements = soup.find_all('dl', class_='glossary')

    # Create a list to store dictionaries
    data_list = []

    # Iterate through each dl element
    for glossary_element in glossary_elements:
        # Find all dt and dd elements within each dl
        dt_elements = glossary_element.find_all('dt', class_='glossary')
        
        # Iterate through each dt element and extract associated dd
        for dt_element in dt_elements:
            dt_text = dt_element.text.strip()
            dd_text = dt_element.find_next('dd', class_='glossary').text.strip()

            # Generate patterns
            patterns = [
                f"what is {dt_text.lower()}",
                dt_text.lower(),
                f"what's {dt_text.lower()}"
            ]

            # Add data to the list
            data_list.append({
                'tag': dt_text.lower(),
                'patterns': patterns,
                'responses': [dd_text]
            })

    # Save the list to a JSON file
    with open('wikioutput.json', 'w', encoding='utf-8') as json_file:
        json.dump(data_list, json_file, indent=2, ensure_ascii=False)

    print("Data saved to 'wikioutput.json'")
else:
    print(f"Failed to retrieve the page. Status code: {response.status_code}")
