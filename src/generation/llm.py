from typing import List

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama

from config import prompts, settings
from src.utils import format_docs, setup_logger

logger = setup_logger(__name__)


class LLMGenerator:
    def __init__(
        self,
        model_name: str,
        temperature: float
    ):
        self.model_name = model_name or settings.OLLAMA_MODEL
        self.temperature = temperature or settings.LLM_TEMPERATURE

        logger.info(
            f"LLMGenerator initialized: model={self.model_name}, "
            f"temperature={self.temperature}"
        )

        # TODO: Initialize Ollama LLM
        self.llm = None

    def get_llm(self) -> Ollama:
        # TODO: Implement lazy loading
        # if self.llm is None:
        #     self.llm = Ollama(
        #         base_url=settings.OLLAMA_BASE_URL,
        #         model=self.model_name,
        #         temperature=self.temperature
        #     )
        # return self.llm

        raise NotImplementedError("TODO: Implement LLM initialization")

    def generate_answer(
        self,
        query: str,
        context_docs: List,
        prompt_template: str
    ) -> str:
        logger.info(f"Generating answer for query: {query[:50]}...")

        # TODO: Implement answer generation
        # Example structure:
        # context = format_docs(context_docs)
        #
        # template = prompt_template or prompts.BASELINE_QA_TEMPLATE
        # prompt = PromptTemplate(
        #     template=template,
        #     input_variables=["context", "question"]
        # )
        #
        # chain = LLMChain(llm=self.get_llm(), prompt=prompt)
        # answer = chain.run(context=context, question=query)
        #
        # logger.info("Answer generated successfully")
        # return answer

        raise NotImplementedError("TODO: Implement answer generation")

    def generate_simple(self, prompt: str) -> str:
        # TODO: Use self.get_llm()(prompt) or similar
        raise NotImplementedError("TODO: Implement simple generation")


if __name__ == "__main__":
    # Test the LLM generator
    generator = LLMGenerator()
    print(f"Model: {generator.model_name}")
    print(f"Temperature: {generator.temperature}")
