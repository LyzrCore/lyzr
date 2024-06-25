"""
Vector store for storing and querying database information, past queries and their associated code.
"""

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
logging.getLogger("chromadb").setLevel(logging.CRITICAL)


class ChromaDBVectorStore:
    """
    A class to manage a vector store using ChromaDB for storing and querying database
    information, DDL statements, past queries and their associated code - SQL, pythonic and plotting code.

    Attributes:
        chroma_client (chromadb.PersistentClient): ChromaDB client instance.
        sql_collection (chromadb.Collection): Collection for storing SQL queries.
        documentation_collection (chromadb.Collection): Collection for storing documentation.
        ddl_collection (chromadb.Collection): Collection for storing DDL statements.
        python_collection (chromadb.Collection): Collection for storing Python code.
        plot_collection (chromadb.Collection): Collection for storing plotting code.
        logger (logging.Logger): Logger instance for logging information and errors.

    Methods:
        __init__(path, remake_store, training_plan, logger):
            Initializes a ChromaDBVectorStore instance.

        make_chroma_client(path):
            Initializes the ChromaDB client and sets up collections for various data types.

        remake_vector_store():
            Remakes the vector store by iterating through existing collections and recreating them if necessary.

        add_training_data(question, sql, ddl, plot_code, python_code, documentation, plan):
            Adds a training plan or individual components to the ChromaDBVectorStore.

        remake_collection(collection_name):
            Remakes a specified collection in the vector store.

        _extract_documents(query_results):
            Extracts and processes documents from the query results.

        get_related_ddl(question):
            Retrieves related DDL queries based on the question.

        get_related_documentation(question):
            Retrieves related documentation based on the question.

        get_related_sql_queries(user_input):
            Retrieves similar question-SQL query pairs based on user input.

        get_related_python_code(question):
            Retrieves similar question-Python code pairs based on the question.

        get_related_plotting_code(question):
            Retrieves similar question-plotting code pairs based on the question.

        generate_embedding(data):
            Generates an embedding for the given data.

        add_question_sql(question, sql):
            Adds a question and its corresponding SQL query to the collection.

        add_ddl(ddl):
            Adds a DDL statement to the collection.

        add_documentation(documentation):
            Adds documentation to the collection.

        add_python_code(question, python_code):
            Adds a question and its corresponding Python code to the collection.

        add_plot_code(question, plot_code):
            Adds a question and its corresponding plotting code to the collection.
    """

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
        training_plan_func: callable = None,  # Function to generate a training plan
        logger: logging.Logger = None,
        **kwargs,
    ):
        """
        Initialize a ChromaDBVectorStore at the specified path.

        Args:
            path (str, optional): Path to the directory where the ChromaDB data is stored. If not provided,
                a MissingValueError is raised.
            remake_store (bool, optional): If True, the store will be recreated. Defaults to False.
            training_plan_func (callable, optional): Function to generate a training plan. Defaults to None. Must return a TrainingPlan object.
            logger (logging.Logger, optional): Logger instance for logging information and errors. Defaults to None.

        Raises:
            MissingValueError: If the 'path' argument is not provided.

        Procedure:
        - If the specified path exists:
            - Initializes the ChromaDB client with the given path.
            - If remake_store is True, attempts to recreate the vector store and logs the process.
            - If a training_plan_func is provided:
                - Calls the function to generate a training plan.
                - Adds this training plan to the vector store.
        - If the specified path does not exist:
            - Creates the directory at the specified path.
            - Initializes the ChromaDB client with the given path.
            - If a training_plan_func is provided:
                - Calls the function to generate a training plan.
                - Adds this training plan to the vector store.

        Example:
            vector_store = ChromaDBVectorStore(
                path="path/to/vector_store",
                remake_store=False,
                training_plan_func=training_plan_func,
                logger=logger,
            )
            docs = vector_store.get_related_documentation(question)
            example_sql = vector_store.get_similar_question_sql(question)
            example_python_code = vector_store.get_similar_python_code(question)
            example_plot_code = vector_store.get_similar_plotting_code(question)
        """
        if path is None:
            raise MissingValueError("path")

        if os.path.exists(path):
            self.make_chroma_client(path=path)
            if remake_store:
                logger.info(f"Remaking vector store at {path}.")
                if not self.remake_vector_store():
                    logger.error(f"Failed to recreate vector store at {path}.")
                if training_plan_func is not None:
                    self.add_training_data(plan=training_plan_func(**kwargs))
            else:
                logger.info(f"Vector store exists at {path}. Using existing store.")
        else:
            os.makedirs(path)
            logger.info(f"Creating vector store at {path}.")
            self.make_chroma_client(path=path)
            if training_plan_func is not None:
                self.add_training_data(plan=training_plan_func(**kwargs))

    def make_chroma_client(self, path: str):
        """
        Initializes the ChromaDB client and sets up collections for various data types.

        Args:
            path (str): The file path where the ChromaDB client will store its data.

        Raises:
            DependencyError: If the required ChromaDB module is not installed.
        """
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
        """
        Remakes the vector store by iterating through existing collections and
        recreating them if necessary. Also ensures that all required collections
        are present in the vector store.

        Returns:
            bool: True if all collections were successfully remade, False otherwise.
        """
        remake = True
        all_collections = vector_store_collections
        existing_collections = self.chroma_client.list_collections()
        for col in existing_collections:
            remake = remake and self.remake_collection(collection_name=col.name)
        for col in all_collections:
            if col in existing_collections:
                continue
            setattr(
                self,
                f"{col}_collection",
                (
                    self.chroma_client.get_or_create_collection(
                        name=col, embedding_function=self.embedding_function
                    )
                ),
            )
        return remake

    def add_training_data(
        self,
        question: str = None,
        sql: str = None,
        ddl: str = None,
        plot_code: str = None,
        python_code: str = None,
        documentation: str = None,
        plan: TrainingPlan = None,
    ) -> str:
        """
        Add a training plan or individual components to the ChromaDBVectorStore.

        Args:
            question (str, optional): The question to be added.
            sql (str, optional): The SQL query associated with the question.
            ddl (str, optional): The Data Definition Language (DDL) statement to be added.
            plot_code (str, optional): The code for generating a plot associated with the question.
            python_code (str, optional): The Python code associated with the question.
            documentation (str, optional): The documentation to be added.
            plan (TrainingPlan, optional): A TrainingPlan object containing multiple items to be added.

        Returns:
            str: A confirmation message or the result of adding the documentation.

        Raises:
            ValidationError: If a question is provided without an associated answer (SQL, plot code, or Python code).
        """
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
        """
        Remakes a specified collection in the vector store.

        Args:
            collection_name (str): The name of the collection to be remade.

        Returns:
            bool: True if the collection was found and remade, False otherwise.

        Procedure:
            - Deletes the existing collection with the given name.
            - Recreates it using the embedding function.
        """
        for collection in vector_store_collections:
            if collection_name == collection:
                self.chroma_client.delete_collection(name=collection)
                setattr(
                    self,
                    f"{collection}_collection",
                    (
                        self.chroma_client.get_or_create_collection(
                            name=collection, embedding_function=self.embedding_function
                        )
                    ),
                )
                return True
        return False

    def _extract_documents(self, query_results) -> list:
        """
        Extracts and processes documents from the query results.

        Args:
            query_results (dict): The results obtained from a query, which may contain a "documents" key.

        Returns:
            list: A list of documents extracted from the query results.
                - If the "documents" key is not present or the query results are None, an empty list is returned.
                - If the documents are in JSON format, they are parsed and returned as a list of dictionaries.
        """
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
        """Retrieve related DDL statements based on the question."""
        return self._extract_documents(
            self.ddl_collection.query(
                query_texts=[question],
            )
        )

    def get_related_documentation(self, question: str) -> list:
        """Retrieve related documentation based on the question."""
        return self._extract_documents(
            self.documentation_collection.query(
                query_texts=[question],
            )
        )

    def get_related_sql_queries(self, user_input: str) -> list:
        """Retrieve similar question - SQL query pairs based on user input."""
        return self._extract_documents(
            self.sql_collection.query(
                query_texts=[user_input],
            )
        )

    def get_related_python_code(self, question: str) -> list:
        """Retrieve similar question - Python code pairs based on the question."""
        return self._extract_documents(
            self.python_collection.query(
                query_texts=[question],
            )
        )

    def get_related_plotting_code(self, question: str) -> list:
        """Retrieve similar question - plotting code pairs based on the question."""
        return self._extract_documents(
            self.plot_collection.query(
                query_texts=[question],
            )
        )

    def generate_embedding(self, data: str) -> list[float]:
        """Generate an embedding for the given data."""
        embedding = self.embedding_function([data])
        if len(embedding) == 1:
            return embedding[0]
        return embedding

    def add_question_sql(self, question: str, sql: str) -> str:
        """Add a question and its corresponding SQL query to the collection."""
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
        """Add a DDL statement to the collection."""
        ddl_id = ChromaDBVectorStore.deterministic_uuid(ddl) + "-ddl"
        self.ddl_collection.add(
            documents=ddl,
            embeddings=self.generate_embedding(ddl),
            ids=ddl_id,
        )
        return ddl_id

    def add_documentation(self, documentation: str) -> str:
        """Add documentation to the collection."""
        doc_id = ChromaDBVectorStore.deterministic_uuid(documentation) + "-doc"
        self.documentation_collection.add(
            documents=documentation,
            embeddings=self.generate_embedding(documentation),
            ids=doc_id,
        )
        return doc_id

    def add_python_code(self, question: str, python_code: str):
        """Add a question and its corresponding Python code to the collection."""
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
        """Add a question and its corresponding plotting code to the collection."""
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
