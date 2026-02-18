from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEndpoint

from config import settings
from src.utils import format_docs, setup_logger

logger = setup_logger(__name__)


class LLMGenerator:
    def __init__(
        self,
        model_name: str | None = None,
        temperature: float | None = None
    ):
        self.model_name: str = model_name or settings.HF_MODEL_NAME
        self.temperature: float = temperature or settings.LLM_TEMPERATURE
        self.llm: HuggingFaceEndpoint | None = None

        logger.info(f"LLMGenerator initialized: model={self.model_name}")

    def get_llm(self) -> HuggingFaceEndpoint:
        if self.llm is None:
            logger.info("Initializing HuggingFace LLM...")
            self.llm = HuggingFaceEndpoint(
                model=self.model_name,
                temperature=self.temperature,
                max_new_tokens=settings.LLM_MAX_TOKENS,
            )
            logger.info("LLM initialized successfully")
        return self.llm

    def build_chain(self, prompt_template: str | None = None):
        prompt = ChatPromptTemplate.from_template(
            prompt_template or """
            You are a helpful assistant answering questions about documentation.
            Use the context below to answer the question.
            If you don't know the answer, say so - don't make it up.

            Context: {context}

            Question: {question}

            Answer:
            """
        )

        chain = prompt | self.get_llm() | StrOutputParser()
        return chain

    def generate_answer(
        self,
        query: str,
        context_docs: list[Document],
        prompt_template: str | None = None
    ) -> str:
        logger.info("Generating answer for query...")

        chain = self.build_chain(prompt_template)

        answer = chain.invoke({
            "context": format_docs(context_docs),
            "question": query
        })

        logger.info("Answer generated successfully")
        return answer.strip()

    def generate_simple(self, prompt: str) -> str:
        chain = ChatPromptTemplate.from_template("{input}") | self.get_llm() | StrOutputParser()
        return chain.invoke({"input": prompt})
