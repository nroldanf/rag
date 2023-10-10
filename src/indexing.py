import os
import time
import warnings
from langchain.vectorstores import Chroma
from langchain.text_splitter import Language
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import LanguageParser
from langchain.text_splitter import RecursiveCharacterTextSplitter

warnings.filterwarnings("ignore")
from utils import process_config_file, get_logger, get_embedding_function, populate_db


logger = get_logger(__name__)

if __name__ == "__main__":
    openai_api_key = os.getenv("OPENAI_API_KEY")

    config = process_config_file("config.conf")
    logger.info(config)

    start = time.perf_counter()
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
    logger.info(f"Documents loaded: {time.perf_counter() - start} seconds")

    start = time.perf_counter()
    logger.info("creating embedding function...")
    embedding_function = get_embedding_function(
        config["embedding"]["type"],
        model_name=config["embedding"]["model_name"],
        cache_dir=config["embedding"]["cache_dir"],
        device=config["embedding"]["device"],
    )
    logger.info(f"Embedding function loaded: {time.perf_counter() - start} seconds")

    logger.info("populating database...")
    start = time.perf_counter()
    db = populate_db(
        config["vectordb"]["type"],
        docs=docs,
        embedding_function=embedding_function,
        persist_directory=config["vectordb"]["persist_directory"],
        collection_name=config["vectordb"]["collection_name"],
    )
    db.persist()
    logger.info(f"Database populated: {time.perf_counter() - start} seconds")
