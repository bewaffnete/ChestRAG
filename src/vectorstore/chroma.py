from langchain_chroma import Chroma
import chromadb

def chroma_instance():
    native_client = chromadb.PersistentClient('/app/chroma_db')
    vectorstore = Chroma(
        client=native_client,
        collection_name='xray',
    )
    return vectorstore

if __name__ == "__main__":
    vectorstore= chroma_instance()
