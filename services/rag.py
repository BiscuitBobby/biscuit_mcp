from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma
from audit import log

def find_similar_content(query: str, text: str, chunk_size: int = 1000, chunk_overlap: int = 0) -> str | None:
    raw_documents = [Document(page_content=text)]

    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    documents = text_splitter.split_documents(raw_documents)

    if not documents:
        log("Warning: No document chunks were created from the input text.")
        return None

    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

    try:
        db = Chroma.from_documents(documents, embedding)
    except Exception as e:
        log(f"Error creating Chroma database: {e}")
        return None

    try:
        docs = db.similarity_search(query)
    except Exception as e:
        log(f"Error during similarity search: {e}")
        return None

    if docs:
        return docs[0].page_content
    else:
        return None
