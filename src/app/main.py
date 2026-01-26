import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import Response

from embedding.model import model_and_processor_instance
from llm.main import llm_instance
from vectorstore.chroma import chroma_instance
from llm.ask import ask_about_photo
from embedding.img2emb import get_image_embedding
from vectorstore.retriever import retriever_func

model, processor, device = model_and_processor_instance()
llm = llm_instance()
vectorstore = chroma_instance()

app = FastAPI(
    title="RAG + Vision API",
    description="API for questions about uploaded photos using RAG",
    version="0.1.0"
)


@app.post("/ask")
async def ask_about_image(
        question: str = Form(...),
        image: UploadFile = File(...)
):
    """
    Accepts an image and a text question.
    Uses RAG to find relevant context and generate a response.
    """
    if not image.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="The uploaded file must be an image."
        )

    try:
        image_bytes = await image.read()
        emb = get_image_embedding(image_bytes, processor, model, device)

        retrieved_docs = retriever_func(emb, vectorstore, k=6)

        answer = ask_about_photo(llm, retrieved_docs, question)

        return Response(
            content=answer,
            media_type="text/markdown; charset=utf-8",
            headers={"X-Content-Type-Options": "nosniff"}
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Request processing error: {str(e)}"
        )


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "vision_model": "loaded",
        "llm": "loaded"
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        workers=1
    )
