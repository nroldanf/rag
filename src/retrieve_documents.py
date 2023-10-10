import os
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA, LLMChain
import mlflow
import time
from utils import process_config_file, get_logger, get_embedding_function
from variables import sensitive_pii, insensitive_pii

logger = get_logger(__name__)

openai_api_key = os.getenv("OPENAI_API_KEY")
config = process_config_file("config.conf")
logger.info(config)

tracking_uri = "http://127.0.0.1:5001"
mlflow.set_tracking_uri(tracking_uri)
experiment = mlflow.set_experiment(experiment_name="zendata")

if __name__ == "__main__":
    # TODO: Make this an entry to the script

    context = "".join(sensitive_pii)
    context = context.join(insensitive_pii)

    query = f"""
    Identify whether not the code below is using any of the following keys within the code and list the ocurrences you find.
    The entities can appear with synonims, and with different capitalization.
    {context}
    """
    start = time.perf_counter()
    embedding_function = get_embedding_function(
        config["embedding"]["type"],
        model_name=config["embedding"]["model_name"],
        cache_dir=config["embedding"]["cache_dir"],
        device=config["embedding"]["device"],
    )
    logger.info(f"Embedding function loaded: {time.perf_counter() - start}")

    db = Chroma(
        persist_directory=config["vectordb"]["persist_directory"],
        embedding_function=embedding_function,
        collection_name=config["vectordb"]["collection_name"],
    )

    docs = db.similarity_search(query=query, k=4, filter={})
