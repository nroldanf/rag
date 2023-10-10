import os
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA, LLMChain
import mlflow
from utils import process_config_file, get_logger

logger = get_logger(__name__)

openai_api_key = os.getenv("OPENAI_API_KEY")
config = process_config_file("config.conf")
logger.info(config)

tracking_uri = "http://127.0.0.1:5001"
mlflow.set_tracking_uri(tracking_uri)
experiment = mlflow.set_experiment(experiment_name="zendata")

if __name__ == "__main__":

    # TODO: Make this an entry to the script
    query = "Check the HTML code and Network data to see if there is info for cookie consent"

    embedding_function = OpenAIEmbeddings(
        chunk_size=config["embedding"]["chunk_size"], openai_api_key=openai_api_key
    )
    db = Chroma(
        persist_directory=config["vectordb"]["persist_directory"],
        embedding_function=embedding_function,
    )
    # retriever = db.as_retriever()

    llm = ChatOpenAI(
        temperature=config["llm"]["temperature"],
        model=config["llm"]["model"],
        openai_api_key=openai_api_key,
    )

    prompt_template = PromptTemplate(
        input_variables=["instruction", "context"],
        template="{instruction}\n\nInput:\n{context}",
    )

    llm_chain = LLMChain(llm=llm, prompt=prompt_template)
    docs = db.similarity_search(query=query, k=2, filter={})

    with mlflow.start_run(experiment_id=experiment.experiment_id):
        context = "".join([doc.page_content for doc in docs])
        result = llm_chain.predict(instruction=query, context=context)
        inputs = [{"question": query, "context": context}]
        mlflow.llm.log_predictions(
            inputs=inputs, outputs=[result], prompts=[prompt_template.template]
        )
        mlflow.log_param("embedding", config["embedding"]["type"])
        mlflow.log_param("vectordb", config["vectodb"]["type"])
        mlflow.log_param("chunk_size", config["embedding"]["chunk_size"])
        mlflow.log_param("llm", config["llm"]["model"])
        mlflow.log_param("temperature", config["llm"]["temperature"])

        logger.info(f"Prediction from LLM: {result}")
