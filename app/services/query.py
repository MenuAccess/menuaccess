import os

import logging

logger = logging.getLogger(__name__)
from app.core.config import settings

os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY

from llama_index import (
    SimpleDirectoryReader,
    GPTSimpleVectorIndex,
    Document,
    LLMPredictor,
    OpenAIEmbedding,
    GPTTreeIndex,
    ServiceContext,
)

from llama_index.output_parsers import LangchainOutputParser
from llama_index.llm_predictor import StructuredLLMPredictor
from llama_index.composability import ComposableGraph

from langchain.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.output_parsers import (
    PydanticOutputParser,
    StructuredOutputParser,
    ResponseSchema,
)
from pydantic import BaseModel, Field, validator

llm_predictor = StructuredLLMPredictor()

from llama_index import QuestionAnswerPrompt, RefinePrompt
from llama_index.prompts.default_prompts import (
    DEFAULT_TEXT_QA_PROMPT_TMPL,
    DEFAULT_REFINE_PROMPT_TMPL,
)


# Define your desired data structure.
class Joke(BaseModel):
    setup: str = Field(description="question to set up a joke")
    punchline: str = Field(description="answer to resolve the joke")

    # You can add custom validation logic easily with Pydantic.
    @validator("setup")
    def question_ends_with_question_mark(cls, field):
        if field[-1] != "?":
            raise ValueError("Badly formed question!")
        return field


parser = PydanticOutputParser(pydantic_object=Joke)

prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

joke_query = "Tell me a joke."
_input = prompt.format_prompt(query=joke_query)


response_schemas = [
    ResponseSchema(
        name="Question",
        description="",
    ),
    ResponseSchema(
        name="Answer", description="Describes the author's work experience/background."
    ),
]

lc_output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
output_parser = LangchainOutputParser(lc_output_parser)

fmt_qa_tmpl = output_parser.format(DEFAULT_TEXT_QA_PROMPT_TMPL)
fmt_refine_tmpl = output_parser.format(DEFAULT_REFINE_PROMPT_TMPL)

qa_prompt = QuestionAnswerPrompt(fmt_qa_tmpl, output_parser=output_parser)
refine_prompt = RefinePrompt(fmt_refine_tmpl, output_parser=output_parser)


def get_and_query(
    user_id,
    index_storage,
    query,
    response_mode,
    nodes,
    service_context,
    embed_model,
    child_branch_factor,
):
    index: GPTSimpleVectorIndex = index_storage[user_id].get_index_or_throw()
    if isinstance(index, GPTTreeIndex):
        response = index.query(
            _input.to_string(),
            child_branch_factor=child_branch_factor,
            service_context=service_context,
            embed_model=embed_model,
            use_async=True,
        )

        print(response)
    else:
        logger.info("Using simple vector index")
        logger.info("Starting query")
        try:
            response = index.query(
                query,
                response_mode=response_mode,
                service_context=service_context,
                similarity_top_k=nodes,
                text_qa_template=qa_prompt,
                refine_template=refine_prompt,
            )
        except Exception as e:
            logger.error(e)
            raise e
    return response
