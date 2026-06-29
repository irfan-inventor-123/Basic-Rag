from sentence_transformers import SentenceTransformer as st
import chromadb
import os
import json

# ==========================================
# 1. Initialize PERSISTENT Vector Database
# ==========================================
cwd = os.getcwd()

PERSIST_DIR = os.path.join(cwd,"my_vector_db")

chroma_client = chromadb.PersistentClient(path=PERSIST_DIR)

with open(fr"{cwd}\db.json", "r") as file:
    data = json.load(file)
documents=[]
for doc in data:
    documents+=data[doc]
    

# 🌟 CRITICAL CHANGE: Explicitly tell ChromaDB to use Cosine Similarity
collection = chroma_client.get_or_create_collection(
    name="knowledge_base_cosine",
    metadata={"hnsw:space": "cosine"} 
)

# ==========================================
# 2. Define the Ollama Function
# ==========================================

model = st("./local_minilm")

print("Successfully")
def get_embedding(text):
    response = model.encode(text).tolist()
    return response
    
# ==========================================
# 3. Add Documents
# ==========================================


ids = [f"doc_{i}" for i in range(len(documents))]

if collection.count() == 0:
    print(f"Database is empty. Generating embeddings and building COSINE index...")
    embeddings = [get_embedding(doc) for doc in documents]
    
    collection.add(
        documents=documents,
        embeddings=embeddings,
        ids=ids
    )
    print(f"Successfully stored {len(documents)} documents.\n")
else:
    print(f"Database already contains {collection.count()} documents. Skipping ingestion.\n")

# ==========================================
# 4. Perform a Vector Search
# ==========================================
def search(query_text, num_results=2):
    print(f"🔍 Searching for: '{query_text}'")
    query_embedding = get_embedding(query_text)
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=num_results
    )
    
    print("Top matches:")
    for i, doc in enumerate(results['documents'][0]):
        distance = results['distances'][0][i]
        
        # 🌟 Convert "Cosine Distance" to "Cosine Similarity" for easier reading
        similarity = 1 - distance 
        print(f"  {i+1}. {doc}")
        print(f"     -> Cosine Similarity: {similarity:.4f} (Distance: {distance:.4f})")
    print("-" * 50)


# Run test queries
while True:    
    user_query = input("Enter your search query (or 'exit' to quit): ")
    if user_query.lower() == 'exit':
        print("Exiting search. Goodbye!")
        break
    else:
        search(user_query) 
