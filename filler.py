# import os

# path = "./SpaceX.txt"

# with open(path) as f:
#     f.read()

# from langchain_ollama import ChatOllama, ChatResponse
# model = ChatOllama(model="llama2")
from langchain_community.document_loaders import TextLoader, DirectoryLoader
import os

def load_documents(docs_path: str):
    print(f"\nLoading docs form {docs_path}.")

    if not os.path.exists(docs_path): 
        raise FileNotFoundError(f"Directory {docs_path} does not exist.")
    
    loader = DirectoryLoader(
        path = docs_path,
        glob = "*.txt",
        loader_cls = TextLoader
    )

    documents = loader.load()

    if len(documents) == 0:
        raise FileNotFoundError(f"No .txt files found in {docs_path}.")
    
    for i, doc in enumerate(documents[:2]):
        print(f"\nDocument {i+1} out of {len(documents)}:")
        print(f"  Source: {doc.metadata['source']}")
        print(f"  Content length: {len(doc.page_content)} characters")
        print(f"  Content preview: {doc.page_content[:100]}...")
        print(f"  metadata: {doc.metadata}")        

    return documents

print(type(load_documents("docs")[0]))