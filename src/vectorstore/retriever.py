from langchain_core.documents import Document


def retriever_func(query_vector, vectorstore, k=5):
    results = vectorstore._collection.query(
        query_embeddings=[query_vector],
        n_results=k,
        include=["metadatas", "distances"]
    )

    docs = []
    for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
        conclusion = meta.get("conclusion", "")
        if not conclusion:
            conclusion = "The conclusion is missing from the metadata"

        doc = Document(
            page_content=conclusion,
            metadata={
                "source": "chroma_metadata",
                "distance": dist,
                "similarity": 1 - dist if dist is not None else 0.0
            }
        )
        docs.append(doc)

    return docs
