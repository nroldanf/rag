import os
import json
import fnmatch
import logging
import configparser
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import Language
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import LanguageParser


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


# def split_docs(docs, chunk_size: int, chunk_overlap: int):
#     js_splitter = RecursiveCharacterTextSplitter.from_language(
#         language=Language.JS,
#         chunk_size=60,
#         chunk_overlap=0
#     )
#     chunks = js_splitter.split_documents(docs)
#     return chunks
