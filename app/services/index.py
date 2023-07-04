import os
import json
import asyncio
from pathlib import Path
import aiofiles
from collections import defaultdict
from functools import partial
from fastapi import File, UploadFile
from datetime import date
from typing import List
from pydantic import BaseModel, Field, validator

import logging

logger = logging.getLogger(__name__)

from llama_index import (
    SimpleDirectoryReader,
    GPTSimpleVectorIndex,
    Document,
    LLMPredictor,
    OpenAIEmbedding,
    GPTTreeIndex,
    ServiceContext,
)

from langchain.chat_models import ChatOpenAI

from app.core.config import settings
from app.schemas.parsed_output import ParsedOutput
from app.schemas.output import Output
from app.services.query import get_and_query
from app.services.environment_service import EnvService
from app.schemas.output import FormattedOutput


class IndexData:
    def __init__(self):
        self.queryable_index = None
        self.individual_indexes = []

    # A safety check for the future
    def get_index_or_throw(self):
        if not self.queryable():
            raise Exception(
                "An index access was attempted before an index was created. This is a programmer error, please report this to the maintainers."
            )
        return self.queryable_index

    def queryable(self):
        return self.queryable_index is not None

    def has_indexes(self, user_id):
        try:
            return (
                len(os.listdir(EnvService.find_shared_file(f"indexes/{user_id}"))) > 0
            )
        except Exception:
            return False

    def has_search_indexes(self, user_id):
        try:
            return (
                len(
                    os.listdir(EnvService.find_shared_file(f"indexes/{user_id}_search"))
                )
                > 0
            )
        except Exception:
            return False

    def add_index(self, index, user_id, file_name):
        self.individual_indexes.append(index)
        self.queryable_index = index

        # Create a folder called "indexes/{USER_ID}" if it doesn't exist already
        Path(f"{EnvService.save_path()}/indexes/{user_id}").mkdir(
            parents=True, exist_ok=True
        )
        # Save the index to file under the user id
        file = f"{file_name}_{date.today().month}_{date.today().day}"
        # If file is > 93 in length, cut it off to 93
        if len(file) > 93:
            file = file[:93]

        index.save_to_disk(
            EnvService.save_path() / "indexes" / f"{str(user_id)}" / f"{file}.json"
        )

        # Export the index to a search index

    def reset_indexes(self, user_id):
        self.individual_indexes = []
        self.queryable_index = None

        # Delete the user indexes
        try:
            # First, clear all the files inside it
            for file in os.listdir(EnvService.find_shared_file(f"indexes/{user_id}")):
                os.remove(EnvService.find_shared_file(f"indexes/{user_id}/{file}"))
            for file in os.listdir(
                EnvService.find_shared_file(f"indexes/{user_id}_search")
            ):
                os.remove(
                    EnvService.find_shared_file(f"indexes/{user_id}_search/{file}")
                )
        except Exception:
            print("No indexes to delete")


class LlamaIndex:

    """Simple Class to create indexes from a list of documents

    1. Create a list of documents
    2. Create a GPTSimpleVectorIndex
    3. Query the index
    4. Parse the results

    """

    def __init__(self, user_id: str, file_name: str):
        os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY
        self.index_storage = defaultdict(IndexData)
        self.user_id = user_id
        self.loop = asyncio.get_event_loop()
        self.index_name = user_id + file_name
        self.index = GPTSimpleVectorIndex
        self.service_context = ServiceContext.from_defaults(
            llm_predictor=LLMPredictor(
                llm=ChatOpenAI(
                    temperature=0,
                    model_name="gpt-3.5-turbo",
                    openai_api_key=settings.OPENAI_API_KEY,
                    client=None,
                )
            )
        )

        # https://gpt-index.readthedocs.io/en/latest/guides/usage_pattern.html#setting-response-mode  TODO: Research best response mode
        self.response_mode = "default"
        # Depth of the tree, 0 is the root node. 1 is the first level of children, etc. As users can only query the root node, this is set to 0 by default.
        self.nodes = 0

        # https://gpt-index.readthedocs.io/en/latest/guides/usage_pattern.html#setting-embed-model
        self.child_branch_factor = 1

        self.query = settings.MENU_QUERY
        self.results = []

    async def upload_file(self, file: UploadFile = File(...)):
        """Handle a file upload, the file should have the user's ID in the filename."""

        # Create Temporary Directory and File to store the file

        #  check if temp exists
        if os.path.exists("/tmp"):
            print("Temp Exists!")


        async with aiofiles.tempfile.TemporaryDirectory() as temp_path:
            async with aiofiles.tempfile.NamedTemporaryFile(
                suffix=".pdf", dir=temp_path, delete=False
            ) as temp_file:
                print("writing file to", temp_path)
                await temp_file.write(await file.read())
                index = await self.loop.run_in_executor(
                    None,
                    partial(
                        self.index_file,
                        Path(temp_file.name),
                    )
                )

        file_name = file.filename
        self.index_storage[self.user_id].add_index(index, self.user_id, file_name)

    def index_file(self, file_path):
        """Index a file"""
        # print contents of file
        if not os.path.exists(file_path):
            raise FileNotFoundError("File not found")
        documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
        index = GPTSimpleVectorIndex.from_documents(
            documents, service_context=self.service_context
        )
        return index

    def load_documents(self, source_name: str) -> list[Document]:
        """Load documents from a tmp directory"""

        if not os.path.exists("tmp/" + source_name):
            raise FileNotFoundError("File not found")
        else:
            print("File found")

        print("Loading documents...")
        print("Source name: " + source_name)

        self.documents = SimpleDirectoryReader("files").load_data()
        return self.documents

    def create_index(self, path: str):
        """Create an index from a list of documents"""

        logger.info("Creating Index")
        index = GPTSimpleVectorIndex.from_documents(
            documents=self.documents, service_context=self.service_context
        )

        # Change index to a GPTSimpleVectorIndex
        return index

    def parse_results(self, unparsed_string: str) -> list[ParsedOutput]:
        """Parse the results of a query"""
        cleaned_response = unparsed_string[0 : unparsed_string.rfind("}") + 1] + "]"
        result = json.loads(cleaned_response)

        return [ParsedOutput(**r) for r in result]

    async def query_index(
        self, user_id: str, file_name: str, restrictions: str
    ):
        """Query the index"""

        try:
            logger.info("Querying Index")
            embedding_model = OpenAIEmbedding()
            embedding_model.last_token_usage = 0
            self.results = (
                get_and_query(
                    user_id=user_id,
                    index_storage=self.index_storage,
                    query=self.query + restrictions,
                    response_mode=self.response_mode,
                    nodes=self.nodes,
                    service_context=self.service_context,
                    embed_model=embedding_model,
                    child_branch_factor=self.child_branch_factor,
                ),
            )

            print(self.results)

        except Exception as e:
            print(e)
            logger.error("Error Querying Index")
            self.results = Output(response="Error", extra_info=None)

        logger.info("Printing Results")
        print(self.results)

        # format the data into an Array of FormattedOutput

        # Insert the data into the database
        # logger.info("Uploading Data to Supabase")
        # supabase_client.insert_data("output", data=output)

        print(self.results)

        return self.results
