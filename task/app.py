import os

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core import embeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.vectorstores import VectorStore
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import SecretStr

from task._constants import API_KEY, DIAL_URL

SYSTEM_PROMPT = """You are a RAG-powered assistant that assists users with their questions about microwave usage.
            
## Structure of User message:
`RAG CONTEXT` - Retrieved documents relevant to the query.
`USER QUESTION` - The user's actual question.

## Instructions:
- Use information from `RAG CONTEXT` as context when answering the `USER QUESTION`.
- Cite specific sources when using information from the context.
- Answer ONLY based on conversation history and RAG context.
- If no relevant information exists in `RAG CONTEXT` or conversation history, state that you cannot answer the question.
"""

USER_PROMPT = """##RAG CONTEXT:
{context}


##USER QUESTION: 
{query}"""


FAISS_INDEX_FILE = "microwave_faiss_index"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOC_PATH = os.path.join(BASE_DIR, "microwave_manual.txt")
INDEX_DIR = os.path.join(BASE_DIR, FAISS_INDEX_FILE)


class MicrowaveRAG:

    def __init__(self, embeddings: AzureOpenAIEmbeddings, llm_client: AzureChatOpenAI):
        self.llm_client = llm_client
        self.embeddings = embeddings
        self.vector_store = self._setup_vectorstore()

    def _setup_vectorstore(self) -> VectorStore:
        """Initialize the RAG system"""
        print("🔄 Initializing Microwave Manual RAG System...")

        if os.path.exists(INDEX_DIR):
            vector_store = FAISS.load_local(
                folder_path=INDEX_DIR,
                embeddings=self.embeddings,
                allow_dangerous_deserialization=True,
            )
        else:
            vector_store = self._create_new_index()
            print("New vector store created")

        return vector_store

    def _create_new_index(self) -> VectorStore:
        print("📖 Loading text document...")

        if not os.path.exists(DOC_PATH):
            raise RuntimeError(
                f"Document not found at {DOC_PATH}. Ensure 'microwave_manual.txt' exists next to app.py."
            )

        loader = TextLoader(file_path=DOC_PATH)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=300, chunk_overlap=50, separators=["\n\n", "\n", "."]
        )

        chunks = splitter.split_documents(documents)
        vector_store = FAISS.from_documents(chunks, self.embeddings)
        vector_store.save_local(INDEX_DIR)

        return vector_store

    def retrieve_context(self, query: str, k: int = 4, score=0.3) -> str:
        """
        Retrieve the context for a given query.
        Args:
              query (str): The query to retrieve the context for.
              k (int): The number of relevant documents(chunks) to retrieve.
              score (float): The similarity score between documents and query. Range 0.0 to 1.0.
        """
        print(f"{'=' * 100}\n🔍 STEP 1: RETRIEVAL\n{'-' * 100}")
        print(f"Query: '{query}'")
        print(
            f"Searching for top {k} most relevant chunks with similarity score {score}:"
        )

        relevant_docs = self.vector_store.similarity_search_with_relevance_scores(
            query=query, k=k, score_threshold=score
        )

        context_parts = []

        for doc, score in relevant_docs:
            context_parts.append(doc.page_content)
            print(f"\n--- (Relevance Score: {score:.3f}) ---")
            print(f"Content: {doc.page_content}")

        print("=" * 100)
        return "\n\n".join(
            context_parts
        )  # will join all chunks in one string with `\n\n` separator between chunks

    def augment_prompt(self, query: str, context: str) -> str:
        print(f"\n🔗 STEP 2: AUGMENTATION\n{'-' * 100}")

        augmented_prompt = USER_PROMPT.format(context=context, query=query)

        print(f"{augmented_prompt}\n{'=' * 100}")
        return augmented_prompt

    def generate_answer(self, augmented_prompt: str) -> str:
        print(f"\n🤖 STEP 3: GENERATION\n{'-' * 100}")

        messages = [
            SystemMessage(content=USER_PROMPT),
            HumanMessage(content=augmented_prompt),
        ]

        response = self.llm_client.invoke(messages)

        print(f"{response.content}\n{'=' * 100}")
        return response.content


def main(rag: MicrowaveRAG):
    print("🎯 Microwave RAG Assistant")

    while True:
        user_question = input("\n> ").strip()

        if user_question.lower() == "exit":
            print("Exiting the chat. Goodbye!")
            break

        # Retrieval
        context = rag.retrieve_context(user_question, k=10, score=0.3)
        # Augmentation
        augmented_prompt = rag.augment_prompt(user_question, context)
        # Generation
        answer = rag.generate_answer(augmented_prompt)


main(
    MicrowaveRAG(
        embeddings=AzureOpenAIEmbeddings(
            model="text-embedding-3-small-1",
            azure_endpoint=DIAL_URL,
            api_key=SecretStr(API_KEY),
        ),
        llm_client=AzureChatOpenAI(
            temperature=0.0,
            azure_deployment="gpt-4o",
            azure_endpoint=DIAL_URL,
            api_key=SecretStr(API_KEY),
            api_version="",
        ),
    )
)
