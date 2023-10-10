import os
import json
import fnmatch
import logging
import configparser
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import (
    OpenAIEmbeddings,
    HuggingFaceEmbeddings,
    HuggingFaceBgeEmbeddings,
)
from langchain.vectorstores import Qdrant, Chroma
import qdrant_client


def process_config_file(config_file: str) -> dict:
    """Reads the configuration file as a dictionary.

    Args:
        config_file (str): Config file relative path.

    Returns:
        dict: Dictionary containing the configuration included in [default].
    """
    config = configparser.ConfigParser()
    try:
        config.read(config_file)
        config = {s: dict(config.items(s)) for s in config.sections()}
    except Exception as e:
        print(e)
        raise
    return config


def get_logger(name, level=logging.INFO):
    # Create a logger
    logger = logging.getLogger(name)

    # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    logger.setLevel(level)

    # Create a console handler and set the level
    console_handler = logging.StreamHandler()

    # Create a formatter and add it to the handler
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(console_handler)
    return logger


# TODO: make this more flexible to use more
def load_documents(dir_name: str, separators: list = ["/>", ">", "}", ";", "\\"]):
    loader = DirectoryLoader(dir_name, show_progress=True, loader_cls=TextLoader)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000, chunk_overlap=200, separators=separators
    )
    texts = text_splitter.split_documents(docs)
    return texts


def filter_documents(dir_name: str, extension: str) -> list:
    """
    Recursively searches for files with a specified extension in a directory.

    :param str dir_name: The base directory to start the search.
    :param str extension: The file extension to filter for.

    :return: A list of file paths matching the specified extension.
    :rtype: list
    """
    files = []
    pattern = f"*.{extension}"
    for root, _, files in os.walk(dir_name):
        for filename in fnmatch.filter(files, pattern):
            files.append(os.path.join(root, filename))
    return files


def get_embedding_function(
    emb_type: str, model_name: str, cache_dir: str = "../models", device: str = "cpu"
):
    """
    Get an embedding function based on the specified embedding type.

    :param str emb_type: The type of embedding function to use. Options are 'huggingface', 'huggingfacebge', or 'openai'.
    :param str model_name: The name or identifier of the pre-trained model to be used.
    :param str cache_dir: Optional. The directory to cache the downloaded models. Default is ../models.
    :param str device: Optional. The device to run the model on. Default is 'cpu'.

    :return: An embedding function based on the specified type and model.
    :rtype: EmbeddingsBase

    Useful References:
    - HuggingFace: https://huggingface.co/spaces/mteb/leaderboard
    - SentenceTransformer Documentation: https://www.sbert.net/docs/package_reference/SentenceTransformer.html

    .. note::
        The `encode_kwargs` and `model_kwargs` parameters are used to set additional keyword arguments for encoding and model initialization.
        For more details, refer to the SentenceTransformer documentation.

    Useful Ref:
    - https://huggingface.co/spaces/mteb/leaderboard
    - On the encode_kwargs and model_kwargs https://www.sbert.net/docs/package_reference/SentenceTransformer.html
    """
    if emb_type == "huggingface":
        embedding = HuggingFaceEmbeddings(
            model_name=model_name,
            cache_folder=cache_dir,
            model_kwargs={"device": device},
            encode_kwargs={
                "normalize_embeddings": True
            },  # set True to compute cosine similarity
        )
    elif emb_type == "huggingfacebge":
        embedding = HuggingFaceBgeEmbeddings(
            model_name=model_name,
            cache_folder=cache_dir,
            model_kwargs={"device": device},
            encode_kwargs={
                "normalize_embeddings": True
            },  # set True to compute cosine similarity
        )
    elif emb_type == "openai":
        embedding = OpenAIEmbeddings(
            model=model_name,
            show_progress_bar=True,
            max_retries=50,
            request_timeout=60 * 15,
        )
    return embedding


def get_vector_db(
    db_type: str,
    embedding_function,
    collection_name: str,
    persist_directory: str = None,
):
    if db_type == "chromadb":
        db = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding_function,
            collection_name=collection_name,
        )
    elif db_type == "qdrant":
        client = qdrant_client.QdrantClient()
        db = Qdrant(
            client=client,
            location=persist_directory,
            embeddings=embedding_function,
            collection_name=collection_name,
        )
    return db


def populate_db(
    db_type: str,
    docs,
    embedding_function,
    persist_directory: str,
    collection_name: str = "default",
):
    if db_type == "chromadb":
        db = Chroma.from_documents(
            docs,
            embedding_function,
            persist_directory=persist_directory,
            collection_name=collection_name,
        )
    elif db_type == "qdrant":
        db = Qdrant.from_documents(
            docs,
            embedding_function,
            path=persist_directory,
            collection_name=collection_name,
        )
    return db
