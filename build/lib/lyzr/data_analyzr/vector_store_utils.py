# standard library imports
import os
import json
import uuid
import logging

# local imports
from lyzr.base.errors import (
    ValidationError,
    MissingValueError,
    DependencyError,
)
from lyzr.data_analyzr.db_connector import (
    DatabaseConnector,
    TrainingPlanItem,
    TrainingPlan,
)


class ChromaDBVectorStore:
    def __init__(
        self,
        path: str = None,  # Path to the directory where the ChromaDB data is stored
        remake_store: bool = False,  # If True, the store will be recreated
        connector: DatabaseConnector = None,  # Database connector
        logger: logging.Logger = None,
    ):
        if path is None:
            raise MissingValueError("path")

        if os.path.exists(path):
            self.make_chroma_client(path=path)
            if remake_store:
                logger.info(f"Remaking vector store at {path}.")
                if self.remake_vector_store():
                    logger.info("Generating and adding training plan.\n")
                    self.add_training_plan(plan=connector.get_default_training_plan())
                else:
                    logger.error(f"Failed to recreate vector store at {path}.")
            else:
                logger.info(f"Vector store exists at {path}. Using existing store.")
        else:
            os.makedirs(path)
            logger.info(f"Creating vector store at {path}.")
            self.make_chroma_client(path=path)
            logger.info("Generating and adding training plan.\n")
            self.add_training_plan(plan=connector.get_default_training_plan())

    def make_chroma_client(self, path: str):
        try:
            import chromadb
            from chromadb.config import Settings
            from chromadb.utils import embedding_functions
        except ImportError:
            raise DependencyError(
                {
                    "chromadb": "chromadb==0.4.22",
                }
            )

        self.embedding_function = embedding_functions.DefaultEmbeddingFunction()

        self.chroma_client = chromadb.PersistentClient(
            path=path, settings=Settings(anonymized_telemetry=False)
        )
        self.sql_collection = self.chroma_client.get_or_create_collection(
            name="sql", embedding_function=self.embedding_function
        )
        self.documentation_collection = self.chroma_client.get_or_create_collection(
            name="documentation", embedding_function=self.embedding_function
        )
        self.ddl_collection = self.chroma_client.get_or_create_collection(
            name="ddl", embedding_function=self.embedding_function
        )

    def remake_vector_store(self):
        remake = True
        for col in self.chroma_client.list_collections():
            remake = remake and self.remake_collection(collection_name=col.name)
        return remake

    def add_training_plan(
        self,
        question: str = None,
        sql: str = None,
        ddl: str = None,
        documentation: str = None,
        plan: TrainingPlan = None,
    ) -> str:
        if question and not sql:
            raise ValidationError("Please also provide a SQL query")

        if documentation:
            return self.add_documentation(documentation)

        if sql:
            return self.add_question_sql(question=question, sql=sql)

        if ddl:
            return self.add_ddl(ddl)

        if plan:
            for item in plan._plan:
                if item.item_type == TrainingPlanItem.ITEM_TYPE_DDL:
                    self.add_ddl(item.item_value)
                elif item.item_type == TrainingPlanItem.ITEM_TYPE_IS:
                    self.add_documentation(item.item_value)
                elif item.item_type == TrainingPlanItem.ITEM_TYPE_SQL:
                    self.add_question_sql(question=item.item_name, sql=item.item_value)

    def remake_collection(self, collection_name: str) -> bool:
        if collection_name == "sql":
            self.chroma_client.delete_collection(name="sql")
            self.sql_collection = self.chroma_client.get_or_create_collection(
                name="sql", embedding_function=self.embedding_function
            )
            return True
        elif collection_name == "ddl":
            self.chroma_client.delete_collection(name="ddl")
            self.ddl_collection = self.chroma_client.get_or_create_collection(
                name="ddl", embedding_function=self.embedding_function
            )
            return True
        elif collection_name == "documentation":
            self.chroma_client.delete_collection(name="documentation")
            self.documentation_collection = self.chroma_client.get_or_create_collection(
                name="documentation", embedding_function=self.embedding_function
            )
            return True
        else:
            return False

    def _extract_documents(self, query_results) -> list:
        if query_results is None:
            return []

        if "documents" in query_results:
            documents = query_results["documents"]
            if len(documents) == 1 and isinstance(documents[0], list):
                try:
                    documents = [json.loads(doc) for doc in documents[0]]
                except Exception:
                    return documents[0]

            return documents

    def get_related_ddl(self, question: str) -> list:
        return self._extract_documents(
            self.ddl_collection.query(
                query_texts=[question],
            )
        )

    def get_related_documentation(self, question: str) -> list:
        return self._extract_documents(
            self.documentation_collection.query(
                query_texts=[question],
            )
        )

    def get_similar_question_sql(self, user_input: str) -> list:
        return self._extract_documents(
            self.sql_collection.query(
                query_texts=[user_input],
            )
        )

    def generate_embedding(self, data: str) -> list[float]:
        embedding = self.embedding_function([data])
        if len(embedding) == 1:
            return embedding[0]
        return embedding

    def add_question_sql(self, question: str, sql: str) -> str:
        question_sql_json = json.dumps(
            {
                "question": question,
                "sql": sql,
            }
        )
        sql_id = str(uuid.uuid4()) + "-sql"
        self.sql_collection.add(
            documents=question_sql_json,
            embeddings=self.generate_embedding(question_sql_json),
            ids=sql_id,
        )
        return sql_id

    def add_ddl(self, ddl: str) -> str:
        ddl_id = str(uuid.uuid4()) + "-ddl"
        self.ddl_collection.add(
            documents=ddl,
            embeddings=self.generate_embedding(ddl),
            ids=ddl_id,
        )
        return ddl_id

    def add_documentation(self, documentation: str) -> str:
        doc_id = str(uuid.uuid4()) + "-doc"
        self.documentation_collection.add(
            documents=documentation,
            embeddings=self.generate_embedding(documentation),
            ids=doc_id,
        )
        return doc_id
