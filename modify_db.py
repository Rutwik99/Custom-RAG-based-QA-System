from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma

DOCS_PATH = "docs"
CHROMA_PATH = "chroma"

def load_docs_directory():
    docs_loader = PyPDFDirectoryLoader(DOCS_PATH)
    return docs_loader.load()

def split_docs(docs: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(docs)

def get_hf_embedding():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
    )
    return embeddings

def get_chunk_ids(chunks: list[Document]):
    prev_page_id = None
    curr_chunk_index = 0

    for chunk in chunks:
        source, page = chunk.metadata.get("source"), chunk.metadata.get("page")
        curr_page_id = f"{source}:{page}"

        if curr_page_id == prev_page_id:
            curr_chunk_index += 1
        else:
            curr_chunk_index = 0

        chunk_id = f"{curr_page_id}:{curr_chunk_index}"
        chunk.metadata["id"] = chunk_id

        prev_page_id = curr_page_id

    return chunks

def update_chroma(chunks: list[Document]):
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_hf_embedding()
    )
    chunks_with_ids = get_chunk_ids(chunks)

    existing_chunks = db.get(include=[])
    existing_chunk_ids = set(existing_chunks["ids"])
    print(f"Number of existing documents in DB: {len(existing_chunk_ids)}")

    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_chunk_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
    else:
        print("✅ No new documents to add")

if __name__ == "__main__":
    docs = load_docs_directory()
    docs_chunks = split_docs(docs)
    update_chroma(docs_chunks)