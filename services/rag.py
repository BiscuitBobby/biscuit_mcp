from llama_index.core import SimpleDirectoryReader, KnowledgeGraphIndex
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.llms.lmstudio import LMStudio
from llama_index.core import StorageContext
from llama_index.core import Settings
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma
from audit import log

def find_similar_content(query: str, text: str, chunk_size: int = 1000, chunk_overlap: int = 0, embedding=None) -> str | None:
    try:
        log(f"Starting find_similar_content with query: '{query}', text length: {len(text)}, chunk_size: {chunk_size}, chunk_overlap: {chunk_overlap}")

        # Create raw documents
        try:
            raw_documents = [Document(page_content=text)]
            log(f"Created raw document with content length: {len(text)}")
        except Exception as e:
            log(f"Error creating raw document: {e}")
            return None

        # Initialize text splitter
        try:
            text_splitter = CharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            log(f"Initialized text splitter with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
        except Exception as e:
            log(f"Error initializing text splitter: {e}")
            log(f"Parameters - chunk_size type: {type(chunk_size)}, chunk_overlap type: {type(chunk_overlap)}")
            return None

        # Split documents
        try:
            documents = text_splitter.split_documents(raw_documents)
            log(f"Split document into {len(documents)} chunks")
        except Exception as e:
            log(f"Error splitting documents: {e}")
            return None

        if not documents:
            log("Warning: No document chunks were created from the input text.")
            return None

        # Handle embedding parameter
        try:
            if embedding is None:
                log("No embedding provided, initializing HuggingFace embeddings with sentence-transformers/all-MiniLM-L6-v2")
                embedding = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    model_kwargs={'device': 'cpu'}
                )
                log("HuggingFace embeddings initialized successfully")
            else:
                log("Using provided embedding instance")
        except Exception as e:
            log(f"Error initializing embeddings: {e}")
            return None

        # Create Chroma database
        try:
            log("Creating Chroma database from documents")
            db = Chroma.from_documents(documents, embedding)
            log("Chroma database created successfully")
        except Exception as e:
            log(f"Error creating Chroma database: {e}")
            return None

        # Perform similarity search
        try:
            log(f"Performing similarity search for query: '{query}'")
            docs = db.similarity_search(query)
            log(f"Similarity search completed, found {len(docs)} results")
        except Exception as e:
            log(f"Error during similarity search: {e}")
            return None

        # Return results
        try:
            if docs:
                result_content = docs[0].page_content
                log(f"Returning most similar document chunk with length: {len(result_content)}")
                return result_content
            else:
                log("No similar documents found, returning fallback message")
                return "relay to the user that there are no relevant results."
        except Exception as e:
            log(f"Error processing search results: {e}")
            return None

    except Exception as e:
        log(f"Unexpected error in find_similar_content: {e}")
        return None