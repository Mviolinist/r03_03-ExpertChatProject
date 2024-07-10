import openai
import os

client = openai.OpenAI()

api_key = os.environ.get('OPENAI_API_KEY_ABAP_TEST')

if api_key:
    print("API Key found")
else:
    print("API Key not found. Please set it first.")

openai.api_key = api_key

class IntentService():
     def __init__(self):
        pass
     
     def get_intent(self, user_question: str):
         # Odwołanie do punktu końcowego openai.ChatCompletion
         response = client.chat.completions.create(
         model="gpt-3.5-turbo",
         messages=[
               {"role": "user", 
                "content": f"""Extrahieren Sie die Schlüsselwörter aus der folgenden Frage.
                 Antworten Sie nicht, sondern geben Sie nur Schlüsselwörter ein. {user_question}"""}
            ]
         )

         # Wyodrębnienie odpowiedzi
         return (response.choices[0].message.content)