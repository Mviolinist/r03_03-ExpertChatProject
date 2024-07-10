import openai
import os

client = openai.OpenAI()

api_key = os.environ.get('OPENAI_API_KEY_ABAP_TEST')

if api_key:
    print("API Key found")
else:
    print("API Key not found. Please set it first.")

# Setting up the API key for the OpenAI client
openai.api_key = api_key

class ResponseService():
     def __init__(self):
        pass
     
     def generate_response(self, facts, user_question):
         # Odwołanie do punktu końcowego openai.ChatCompletion
         response = client.chat.completions.create(
         model="gpt-3.5-turbo",
         messages=[
               {"role": "user", "content": f"""Uwzględniając FAKTY odpowiedz na PYTANIE.
                PYTANIE: {user_question}. FAKTY: {facts}"""}
            ]
         )

         # Wyodrębnienie odpowiedzi
         return (response.choices[0].message.content)