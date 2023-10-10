import os
import warnings
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import Language
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import LanguageParser
from langchain.text_splitter import RecursiveCharacterTextSplitter

warnings.filterwarnings("ignore")
from utils import process_config_file, get_logger, load_documents, filter_documents


logger = get_logger(__name__)

if __name__ == "__main__":
    openai_api_key = os.getenv("OPENAI_API_KEY")

    config = process_config_file("config.conf")
    logger.info(config)

    logger.info("Loading documents...")
    loader = GenericLoader.from_filesystem(
        path=config["data"]["documents_directory"],
        # glob="*",
        suffixes=[".js"],
        show_progress=True,
        parser=LanguageParser(Language.JS),
    )
    full_docs = loader.load()
    js_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.JS,
        chunk_size=int(config["embedding"]["chunk_size"]),
        chunk_overlap=int(config["embedding"]["chunk_overlap"]),
    )
    docs = js_splitter.split_documents(full_docs)
    logger.info(docs[0])

    embeddings = OpenAIEmbeddings(
        show_progress_bar=True, chunk_size=4, openai_api_key=openai_api_key
    )
    db = Chroma(
        persist_directory=config["vectodb"]["persist_directory"],
        embedding_function=embeddings,
    )
    db = Chroma.from_documents(
        docs, embeddings, persist_directory=config["vectodb"]["persist_directory"]
    )
    db.persist()
