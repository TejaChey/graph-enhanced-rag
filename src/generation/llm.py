from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.runnables import RunnablePassthrough

# --- Previous model: HuggingFace Inference Provider (paid quota, Together.ai router) ---
# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# --- Current model: local HuggingFace pipeline (free, no API credits needed) ---
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

from config import settings
from src.utils import format_docs


class LLMGenerator:
    def __init__(self, model_name=None, temperature=None):
        self.model_name = model_name or settings.HF_MODEL_NAME
        self.temperature = temperature if temperature is not None else settings.LLM_TEMPERATURE
        self._llm = None

    def get_llm(self):
        if self._llm is None:
            # --- Previous: paid Inference Provider endpoint ---
            # endpoint = HuggingFaceEndpoint(
            #     model=self.model_name,          # e.g. "Qwen/Qwen2.5-7B-Instruct"
            #     temperature=self.temperature,
            #     max_new_tokens=settings.LLM_MAX_TOKENS,
            # )
            # self._llm = ChatHuggingFace(llm=endpoint)

            # --- Current: local pipeline (google/flan-t5-base) ---
            model_name = "google/flan-t5-base"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            pipe = pipeline(
                "text2text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=settings.LLM_MAX_TOKENS,
            )
            self._llm = HuggingFacePipeline(pipeline=pipe)
        return self._llm

    def build_rag_chain(self, retriever, prompt_template=None):
        template = prompt_template or (
            "Answer the following question using ONLY the context below.\n"
            "If the answer is not in the context, say you don't know.\n\n"
            "Context:\n{context}\n\n"
            "Question: {input}\n\n"
            "Answer:"
        )

        prompt = PromptTemplate(input_variables=["context", "input"], template=template)

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
        template = prompt_template or (
            "Answer the following question using ONLY the context below.\n"
            "If the answer is not in the context, say you don't know.\n\n"
            "Context:\n{context}\n\n"
            "Question: {input}\n\n"
            "Answer:"
        )

        prompt = PromptTemplate(input_variables=["context", "input"], template=template)
        chain = prompt | self.get_llm() | StrOutputParser()
        return chain.invoke({
            "context": format_docs(context_docs),
            "input": query,
        }).strip()

    def generate_simple(self, prompt_text):
        prompt = PromptTemplate(input_variables=["input"], template="{input}")
        chain = prompt | self.get_llm() | StrOutputParser()
        return chain.invoke({"input": prompt_text})
