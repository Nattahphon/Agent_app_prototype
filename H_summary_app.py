import os
import logging
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import pandas as pd



load_dotenv()


class SummaryAgent():
    """A class to summarize the outputs of the PandasAgent."""

    def __init__(self, temperature: float, base_url: str, model_name: str):
        self.llm = self.initialize_llm(temperature, base_url, model_name)

    @staticmethod
    def initialize_llm(temperature: float, base_url: str, model_name: str) -> ChatOpenAI:
        """Initialize the LLM for summarization."""
        api_key = os.getenv("PLOT_API_KEY")
        if not api_key:
            raise ValueError("API key is missing. Ensure 'API_KEY' is set in your environment.")
        return ChatOpenAI(
            base_url=base_url,
            model=model_name,
            api_key=api_key,
            temperature=temperature,
        )

    def summarize(self, text: str) -> str:
        parser = JsonOutputParser()

        """Generate a summary of the given text."""
        # Create the prompt using `PromptTemplate` and render it with the input text
        templete = ("""
        You are a meticulous research process note-taker. Your main responsibility is to observe, summarize, and document the actions and findings of the research team. Your tasks include:
        Summarize the following content in a concise and clear manner:\n{text}
        1. Observing and recording key activities, decisions, and discussions among team members.
        2. Summarizing complex information into clear, concise, and accurate notes.
        3. Organizing notes in a structured format that ensures easy retrieval and reference.
        4. Highlighting significant insights, breakthroughs, challenges, or any deviations from the research plan.
        5. Responding only in {format_instructions}.

        Your output should be well-organized and easy to integrate with other project documentation.
        """)
        
        prompt = PromptTemplate(
            template=templete, 
            input_variables=['text'],
            partial_variables={'format_instructions': parser.get_format_instructions()}, 
            )  

        try:
            chain  = prompt | self.llm | parser
            response = chain.invoke({'text' : text})
            return response
        except Exception as e:
            logging.error(f"Error during summarization: {e}")
            return "An error occurred while generating the summary."
        
    

        




if __name__ == "__main__":
    sum_agent = SummaryAgent(
        temperature=0.1,
        base_url="https://api.opentyphoon.ai/v1",
        model_name="typhoon-v1.5x-70b-instruct",)
    

    text: str = """What is Lorem Ipsum?
Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum."""
    data = sum_agent.summarize(text=text)
    # Decomposing and printing components
    print("Summary:")
    print(data['summary'])
    print("\nNotes:")
    for note in data['notes']:
        for key, value in note.items():
            print(f"  {key.capitalize()}: {value}")

    print("\nHighlights:")
    for key, value in data['highlights'].items():
        print(f"  {key.capitalize()}: {value}")

