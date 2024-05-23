from dotenv import load_dotenv

load_dotenv()
from intentservice import IntentService
from responseservice import ResponseService
from dataservice import DataService

# Przykładowy plik PDF
pdf = 'tiere.pdf'
#pdf = 'C:\\Users\\Michał\\Desktop\\Tworzenie aplikacji z GPT\\r03_03 Ekspert od Minecraft\\Minecraft.pdf'

data_service = DataService()

# Usunięcie wszystkich danych z bazy Redis
data_service.drop_redis_data()

# Załadowanie danych z dokumentu PDF do bazy Redis
data = data_service.pdf_to_embeddings(pdf)

data_service.load_data_to_redis(data)

intent_service = IntentService()
response_service = ResponseService()

# Pytanie
question = 'Welche Tiere können lachen?'
# Określenie intencji
intents = intent_service.get_intent(question)
# Uzyskanie faktów
facts = data_service.search_redis(intents)
# Udzielenie odpowiedzi
answer = response_service.generate_response(facts, question)
print(answer)