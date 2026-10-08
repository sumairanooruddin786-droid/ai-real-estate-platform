"""One command that prepares everything the web app needs:

    python run_pipeline.py

Steps: clean data + charts  ->  train and save the model  ->  build the knowledge guide  ->  index documents.
"""
from src import build_knowledge, eda, rag, train_model

if __name__ == "__main__":
    print("1/4  Cleaning data and drawing charts")
    eda.run()
    print("\n2/4  Training and comparing models")
    train_model.run()
    print("\n3/4  Building the neighborhood guide")
    build_knowledge.run()
    print("\n4/4  Indexing documents for the assistant")
    service = rag.RAGService()
    service.ingest_file(rag.KNOWLEDGE_DIR / "05_area_price_guide.md")  # refresh after a re-run
    print(f"Search backend: {service.backend}")
    for d in service.documents():
        print(f"  {d['name']}: {d['chunks']} chunks")
    print("\nDone. Start the app with:  uvicorn app.main:app --reload")
