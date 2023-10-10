# https://docs.llamaindex.ai/en/stable/api_reference/node_postprocessor.html

from llama_index.indices.postprocessor import (
    PIINodePostprocessor,
    NERPIINodePostprocessor,
)
from llama_index.llms import HuggingFaceLLM
from llama_index import ServiceContext, Document, VectorStoreIndex
from llama_index.schema import TextNode
from llama_index.schema import NodeWithScore
from utils import get_logger

logger = get_logger(__name__)


if __name__ == "__main__":
    file_name = (
        "../docs/pii_data/nytimes/raw_data/7bc8bccf5c254286a99b11c68f6bf4ce.min.js"
    )
    with open(file_name, "r") as f:
        text = f.read()

    # TODO:
    processor = "NER"

    node = TextNode(text=text)
    # Using huggingface
    service_context = ServiceContext.from_defaults()

    if processor == "NER":
        processor = NERPIINodePostprocessor(service_context=service_context)
    elif processor == "LLM":
        processor = PIINodePostprocessor(service_context=service_context)

    new_nodes = processor.postprocess_nodes([NodeWithScore(node=node)])
    logger.info(new_nodes[0].node.get_text())
    logger.info(new_nodes[0].node.metadata["__pii_node_info__"])
