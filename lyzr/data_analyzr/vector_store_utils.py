# standard library imports
import os
import json
import logging

# local imports
from lyzr.base.errors import (
    ValidationError,
    MissingValueError,
    DependencyError,
)

vector_store_collections = ["sql", "ddl", "documentation", "python", "plot"]


class ChromaDBVectorStore:
    from lyzr.data_analyzr.utils import deterministic_uuid
    from lyzr.data_analyzr.db_connector import (
        TrainingPlanItem,
        DatabaseConnector,
        DatabaseConnector,
        TrainingPlanItem,
        DatabaseConnector,
        TrainingPlanItem,
        TrainingPlan,
    )

    def __init__(
        self,
        path: str = None,  # Path to the directory where the ChromaDB data is stored
        remake_store: bool = False,  # If True, the store will be recreated
        training_plan: TrainingPlan = None,  # Training plan
        logger: logging.Logger = None,
    ):
        if path is None:
            raise MissingValueError("path")

        if os.path.exists(path):
            self.make_chroma_client(path=path)
            if remake_store:
                logger.info(f"Remaking vector store at {path}.")
                if not self.remake_vector_store():
                    logger.error(f"Failed to recreate vector store at {path}.")
                if training_plan is not None:
                    self.add_training_plan(plan=training_plan)
            else:
                logger.info(f"Vector store exists at {path}. Using existing store.")
        else:
            os.makedirs(path)
            logger.info(f"Creating vector store at {path}.")
            self.make_chroma_client(path=path)
            if training_plan is not None:
                self.add_training_plan(plan=training_plan)

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
        self.python_collection = self.chroma_client.get_or_create_collection(
            name="python", embedding_function=self.embedding_function
        )
        self.plot_collection = self.chroma_client.get_or_create_collection(
            name="plot", embedding_function=self.embedding_function
        )

    def remake_vector_store(self):
        remake = True
        all_collections = vector_store_collections
        existing_collections = self.chroma_client.list_collections()
        for col in existing_collections:
            remake = remake and self.remake_collection(collection_name=col.name)
        for col in all_collections:
            if col in existing_collections:
                continue
            self.__dict__[f"{col}_collection"] = (
                self.chroma_client.get_or_create_collection(
                    name=col, embedding_function=self.embedding_function
                )
            )
        return remake

    def add_training_plan(
        self,
        question: str = None,
        sql: str = None,
        ddl: str = None,
        plot_code: str = None,
        python_code: str = None,
        documentation: str = None,
        plan: TrainingPlan = None,
    ) -> str:
        if (
            (question and not sql)
            and (question and not plot_code)
            and (question and not python_code)
        ):
            raise ValidationError("Please also provide an answer to the question.")

        if documentation:
            return self.add_documentation(documentation)

        if sql:
            return self.add_question_sql(question=question, sql=sql)

        if ddl:
            return self.add_ddl(ddl)

        if plot_code:
            self.add_plot_code(question=question, plot_code=plot_code)

        if python_code:
            self.add_python_code(question=question, python_code=python_code)

        if plan:
            for item in plan._plan:
                if item.item_type == ChromaDBVectorStore.TrainingPlanItem.ITEM_TYPE_DDL:
                    self.add_ddl(item.item_value)
                elif (
                    item.item_type == ChromaDBVectorStore.TrainingPlanItem.ITEM_TYPE_IS
                ):
                    self.add_documentation(item.item_value)
                elif (
                    item.item_type == ChromaDBVectorStore.TrainingPlanItem.ITEM_TYPE_SQL
                ):
                    self.add_question_sql(question=item.item_name, sql=item.item_value)
                elif (
                    item.item_type == ChromaDBVectorStore.TrainingPlanItem.ITEM_TYPE_PY
                ):
                    self.add_python_code(
                        question=item.item_name, python_code=item.item_value
                    )
                elif (
                    item.item_type
                    == ChromaDBVectorStore.TrainingPlanItem.ITEM_TYPE_PLOT
                ):
                    self.add_plot_code(
                        question=item.item_name, plot_code=item.item_value
                    )

    def remake_collection(self, collection_name: str) -> bool:
        for collection in vector_store_collections:
            if collection_name == collection:
                self.chroma_client.delete_collection(name=collection)
                self.__dict__[f"{collection}_collection"] = (
                    self.chroma_client.get_or_create_collection(
                        name=collection, embedding_function=self.embedding_function
                    )
                )
                return True
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

    def get_similar_python_code(self, question: str) -> list:
        return self._extract_documents(
            self.python_collection.query(
                query_texts=[question],
            )
        )

    def get_similar_plotting_code(self, question: str) -> list:
        return self._extract_documents(
            self.plot_collection.query(
                query_texts=[question],
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
        sql_id = ChromaDBVectorStore.deterministic_uuid(question_sql_json) + "-sql"
        self.sql_collection.add(
            documents=question_sql_json,
            embeddings=self.generate_embedding(question_sql_json),
            ids=sql_id,
        )
        return sql_id

    def add_ddl(self, ddl: str) -> str:
        ddl_id = ChromaDBVectorStore.deterministic_uuid(ddl) + "-ddl"
        self.ddl_collection.add(
            documents=ddl,
            embeddings=self.generate_embedding(ddl),
            ids=ddl_id,
        )
        return ddl_id

    def add_documentation(self, documentation: str) -> str:
        doc_id = ChromaDBVectorStore.deterministic_uuid(documentation) + "-doc"
        self.documentation_collection.add(
            documents=documentation,
            embeddings=self.generate_embedding(documentation),
            ids=doc_id,
        )
        return doc_id

    def add_python_code(self, question: str, python_code: str):
        question_analysis_steps = json.dumps(
            {
                "question": question,
                "python_code": python_code,
            }
        )
        python_id = (
            ChromaDBVectorStore.deterministic_uuid(question_analysis_steps) + "-python"
        )
        self.python_collection.add(
            documents=question_analysis_steps,
            embeddings=self.generate_embedding(question_analysis_steps),
            ids=python_id,
        )
        return python_id

    def add_plot_code(self, question: str, plot_code: str):
        question_plot_steps = json.dumps(
            {
                "question": question,
                "plot_code": plot_code,
            }
        )
        code_id = ChromaDBVectorStore.deterministic_uuid(question_plot_steps) + "-code"
        self.plot_collection.add(
            documents=question_plot_steps,
            embeddings=self.generate_embedding(question_plot_steps),
            ids=code_id,
        )
        return code_id
