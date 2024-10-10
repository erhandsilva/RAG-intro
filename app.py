import os
from dotenv import load_dotenv
from anthropic import Anthropic
from chromadb.utils import embedding_functions
import voyageai

#load environment variables from .env files
load_dotenv()

anthropic_key = os.environ.get('ANTHROPIC_API_KEY')
voyageai_key  = os.environ.get('VOYAGE_API_KEY')

# initiliaze the Voyage Ai client
vo = voyageai.Client(api_key=voyageai_key)

texts = ["Sample text 1", "Sample text 2"]

result = vo.embed(texts, model="voyage-2", input_type="document")

print (result.embeddings[0])
print (result.embeddings[1])


