import os
import hashlib
from pathlib import Path
from gemini_summarise import summarize
from langchain_openai import ChatOpenAI
from langchain_postgres import PGVector
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader

# ----------------------- SETUP -----------------------
local_llm = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
connection = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"

summary_vector_store = PGVector(
    embeddings=embeddings,
    collection_name="case_summaries",
    connection=connection,
    use_jsonb=True,
)

case_vector_store = PGVector(
    embeddings=embeddings,
    collection_name="case_docs",
    connection=connection,
    use_jsonb=True,
)

# ----------------------- PDF PROCESSING -----------------------
def extract_pdf_content(file_path: Path) -> str:
    try:
        loader = PyPDFLoader(str(file_path))
        pages = loader.load()
        return "\n\n".join([page.page_content for page in pages])
    except Exception:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except:
            return ""

def compute_file_hash(file_path: Path) -> str:
    with open(file_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

# ----------------------- INGESTION -----------------------
def is_file_already_ingested(file_id: str, file_hash: str) -> bool:
    try:
        for store in [summary_vector_store, case_vector_store]:
            results = store.similarity_search(query="", k=1, filter={"id": file_id})
            if results:
                stored_hash = results[0].metadata.get("file_hash")
                if stored_hash == file_hash:
                    return True
                if stored_hash:
                    print(f"⚠️ File {file_id} exists but hash differs.")
                    return False
        return False
    except Exception as e:
        print(f"Error checking file status: {e}")
        return False

def ingest_documents(folder_path: str, max_docs: int = 5):
    folder = Path(folder_path)
    summary_docs, case_docs = [], []
    processed, skipped = [], []

    for i, file in enumerate(folder.iterdir()):
        if not file.is_file() or i >= max_docs:
            continue

        file_hash = compute_file_hash(file)
        if is_file_already_ingested(file.name, file_hash):
            print(f"⏭️  Skipped: {file.name}")
            skipped.append(file.name)
            max_docs+=1
            continue

        print(f"📄 Ingesting: {file.name}")
        summary = summarize(file, model="gemini-2.5-flash")
        content = extract_pdf_content(file)
        metadata = {
            "id": file.name,
            "location": str(file),
            "file_hash": file_hash,
            "file_size": file.stat().st_size,
            "file_type": file.suffix.lower()
        }

        summary_docs.append(Document(page_content=summary, metadata=metadata))
        case_docs.append(Document(page_content=content, metadata={**metadata, "content_type": "full_document"}))
        processed.append(file.name)

    if summary_docs:
        summary_vector_store.add_documents(summary_docs)
        print(f"✅ Added {len(summary_docs)} summary documents.")

    if case_docs:
        case_vector_store.add_documents(case_docs)
        print(f"✅ Added {len(case_docs)} full documents.")

    print("\n📊 Ingestion Summary:")
    print(f"  ✅ New: {len(processed)}")
    print(f"  ⏭️ Skipped: {len(skipped)}")

# ----------------------- RETRIEVAL -----------------------
def search_law_documents(query: str, summary_top_k: int = 5, doc_top_k: int = 10, doc_score_threshold: float = 0.8):
    print("=" * 80)
    print("🔍 SUMMARY RESULTS")
    print("=" * 80)

    summary_results = summary_vector_store.similarity_search_with_score(query, k=summary_top_k)
    for i, (doc, score) in enumerate(summary_results, 1):
        print(f"\nSummary {i} (Score: {score:.4f})")
        print(f"File: {doc.metadata.get('id')}")
        print(doc.page_content[:400] + "...")
        print("-" * 60)

    print("\n" + "=" * 80)
    print("📘 FULL DOCUMENT RESULTS")
    print("=" * 80)

    case_results = case_vector_store.similarity_search_with_score(query, k=doc_top_k)
    filtered_results = [(doc, score) for doc, score in case_results if score < doc_score_threshold]

    for i, (doc, score) in enumerate(filtered_results, 1):
        print(f"\nDoc {i} (Score: {score:.4f})")
        print(f"File: {doc.metadata.get('id')}")
        print(doc.page_content[:400] + "...")
        print("-" * 60)

# ----------------------- MAIN -----------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Law Retrieval Agent")
    parser.add_argument("--ingest", type=str, help="Path to folder containing legal documents")
    parser.add_argument("--query", type=str, help="Search query")
    parser.add_argument("--max_docs", type=int, default=5, help="Max number of documents to ingest")
    args = parser.parse_args()

    if args.ingest:
        ingest_documents(args.ingest, max_docs=args.max_docs)

    if args.query:
        search_law_documents(args.query)
