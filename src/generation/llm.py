from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEndpoint

from config import settings
from src.utils import format_docs


class LLMGenerator:
    def __init__(self, model_name=None, temperature=None):
        self.model_name = model_name or settings.HF_MODEL_NAME
        self.temperature = temperature if temperature is not None else settings.LLM_TEMPERATURE
        self._llm = None

    def get_llm(self):
        if self._llm is None:
            self._llm = HuggingFaceEndpoint(
                model=self.model_name,
                temperature=self.temperature,
                max_new_tokens=settings.LLM_MAX_TOKENS,
            )
        return self._llm

    def build_rag_chain(self, retriever, prompt_template=None):
        """
        Build a modern agentic RAG chain using pure LCEL.
        Accepts {"input": question} and produces {"input", "context", "answer"}.
        """
        system_prompt = prompt_template or (
            "You are a helpful assistant answering questions about documentation. "
            "Use the following retrieved context to answer the question. "
            "If you don't know the answer based on the context, say so — don't make it up.\n\n"
            "Context:\n{context}"
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])

        return (
            RunnablePassthrough.assign(
                context=lambda x: format_docs(retriever.invoke(x["input"]))
            )
            | RunnablePassthrough.assign(
                answer=prompt | self.get_llm() | StrOutputParser()
            )
        )

    def build_conversational_rag_chain(self, retriever):
        """
        Build a conversational agentic RAG chain with chat history support.
        Accepts {"input": question, "chat_history": [messages]}.
        """
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "Given the chat history and the latest user question, "
             "reformulate the question to be standalone and self-contained. "
             "Do NOT answer — just rewrite if needed, otherwise return as-is."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        history_aware_retriever = (
            RunnablePassthrough.assign(
                standalone_question=contextualize_q_prompt | self.get_llm() | StrOutputParser()
            )
            | RunnablePassthrough.assign(
                context=lambda x: format_docs(retriever.invoke(x["standalone_question"]))
            )
        )

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a helpful assistant answering questions about documentation. "
             "Use the context below to answer. If unsure, say so.\n\nContext:\n{context}"),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        return history_aware_retriever | RunnablePassthrough.assign(
            answer=qa_prompt | self.get_llm() | StrOutputParser()
        )

    def generate_answer(self, query, context_docs, prompt_template=None):
        """
        Generate an answer directly from a list of pre-retrieved documents.
        Useful when retrieval is handled externally (e.g. graph retrieval).
        """
        system = prompt_template or (
            "You are a helpful assistant answering questions about documentation. "
            "Use the context below to answer. If unsure, say so.\n\nContext:\n{context}"
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system),
            ("human", "{input}"),
        ])

        chain = prompt | self.get_llm() | StrOutputParser()
        return chain.invoke({
            "context": format_docs(context_docs),
            "input": query,
        }).strip()

    def generate_simple(self, prompt):
        chain = ChatPromptTemplate.from_template("{input}") | self.get_llm() | StrOutputParser()
        return chain.invoke({"input": prompt})
