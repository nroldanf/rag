import os
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA, LLMChain
from langchain.text_splitter import Language
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import LanguageParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
import mlflow
from utils import process_config_file, get_logger, filter_documents
from variables import sensitive_pii, insensitive_pii

logger = get_logger(__name__)

openai_api_key = os.getenv("OPENAI_API_KEY")
config = process_config_file("config.conf")
logger.info(config)

tracking_uri = "http://127.0.0.1:5002"
mlflow.set_tracking_uri(tracking_uri)
experiment = mlflow.set_experiment(experiment_name="zendata")

if __name__ == "__main__":
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

    for text in docs:
        query = f"""
        Identify whether not the code below is using any of the following keys within the code and list the ocurrences you find.
        The entities can appear with synonims, and with different capitalization.
        {text}
        """

        llm = ChatOpenAI(
            temperature=config["llm"]["temperature"],
            model=config["llm"]["model"],
            openai_api_key=openai_api_key,
        )

        prompt_template = PromptTemplate(
            input_variables=["instruction", "context"],
            template="{instruction}\n\The categories are the following:\n{context}. Explain you reasoning.",
        )

        llm_chain = LLMChain(llm=llm, prompt=prompt_template)
        # docs = db.similarity_search(query=query, k=2, filter={})

        with mlflow.start_run(experiment_id=experiment.experiment_id):
            # context = "".join([doc.page_content for doc in docs])
            context = "".join(sensitive_pii)
            context = context.join(insensitive_pii)
            result = llm_chain.predict(instruction=query, context=context)
            inputs = [{"question": query, "context": context}]
            mlflow.llm.log_predictions(
                inputs=inputs, outputs=[result], prompts=[prompt_template.template]
            )
            mlflow.log_param("llm", config["llm"]["model"])
            mlflow.log_param("temperature", config["llm"]["temperature"])
            mlflow.log_param("file_name", text.metadata["source"].split("/")[-1])
            mlflow.log_param("webpage", text.metadata["source"].split("/")[3])

            logger.info(f"Prediction from LLM: {result}")
