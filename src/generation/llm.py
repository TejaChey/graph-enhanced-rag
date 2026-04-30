from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.runnables import RunnablePassthrough

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFacePipeline
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

from config import settings
from src.utils import format_docs


class LLMGenerator:
    def __init__(self, model_name=None, temperature=None, use_local=False):
        self.use_local = use_local
        if self.use_local:
            self.model_name = model_name or "google/flan-t5-base"
        else:
            self.model_name = model_name or settings.HF_MODEL_NAME
        self.temperature = temperature if temperature is not None else settings.LLM_TEMPERATURE
        self._llm = None

    def get_llm(self):
        if self._llm is None:
            if self.use_local:
                tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
                pipe = pipeline(
                    "text2text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    max_new_tokens=settings.LLM_MAX_TOKENS,
                )
                self._llm = HuggingFacePipeline(pipeline=pipe)
            else:
                endpoint = HuggingFaceEndpoint(
                    model=self.model_name,
                    temperature=self.temperature,
                    max_new_tokens=settings.LLM_MAX_TOKENS,
                )
                self._llm = ChatHuggingFace(llm=endpoint)
        return self._llm

    def build_rag_chain(self, retriever, prompt_template=None):
        system_prompt = prompt_template or (
            "You are a helpful assistant answering questions about documentation. "
            "Use the following retrieved context to answer the question. "
            "If you don't know the answer based on the context, say so — don't make it up.\n\n"
            "Context:\n{context}"
        )

        if self.use_local:
            template = system_prompt + "\n\nQuestion: {input}\n\nAnswer:"
            prompt = PromptTemplate(input_variables=["context", "input"], template=template)
        else:
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
        system = prompt_template or (
            "You are a helpful assistant answering questions about documentation. "
            "Use the context below to answer. If unsure, say so.\n\nContext:\n{context}"
        )

        if self.use_local:
            template = system + "\n\nQuestion: {input}\n\nAnswer:"
            prompt = PromptTemplate(input_variables=["context", "input"], template=template)
        else:
            prompt = ChatPromptTemplate.from_messages([
                ("system", system),
                ("human", "{input}"),
            ])
        chain = prompt | self.get_llm() | StrOutputParser()
        return chain.invoke({
            "context": format_docs(context_docs),
            "input": query,
        }).strip()

    def generate_simple(self, prompt_text):
        if self.use_local:
            prompt = PromptTemplate(input_variables=["input"], template="{input}")
            chain = prompt | self.get_llm() | StrOutputParser()
        else:
            chain = ChatPromptTemplate.from_template("{input}") | self.get_llm() | StrOutputParser()
        return chain.invoke({"input": prompt_text})
