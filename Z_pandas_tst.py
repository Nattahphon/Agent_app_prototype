import os
import pandas as pd
import logging
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
from H_datahandle_app import DataHandler
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import json
import numpy as np
import matplotlib as plt
from langchain.output_parsers import OutputFixingParser
import re

# First, create a Pydantic model for your output structure
class PlotResponse(BaseModel):
    query: str = Field(description="Description of what the code does")
    explanation: str = Field(description="Detailed explanation of the analysis")
    code: str = Field(description="The python code")

class PandasAgent:
    def __init__(self, temperature: float, base_url: str, model_name: str, dataset_paths: dict):
        self.handler = DataHandler(dataset_paths=dataset_paths)
        self.handler.load_data()
        self.handler.preprocess_data()

        self.temperature = temperature
        self.base_url = base_url
        self.model_name = model_name
        self.api_key = os.getenv("PANDAS_API_KEY")
        self.llm = self.initialize_llm()
        self.output_parser = PydanticOutputParser(pydantic_object=PlotResponse)


    def initialize_llm(self) -> ChatOpenAI:
        """Initialize the language model."""
        if not self.api_key:
            raise ValueError("API key is missing. Ensure 'PANDAS_API_KEY' is set in your environment.")
        return ChatOpenAI(
            base_url=self.base_url,
            model=self.model_name,
            api_key=self.api_key,
            temperature=self.temperature,
        )

    def create_agent(self, df_key: str):
        """Create an agent for the specified dataset."""
        if df_key not in self.handler._data:
            raise ValueError(f"Dataset '{df_key}' not found.")
        
        df = self.handler.get_data(df_key)

        # Prefix: บริบททั่วไปที่กำหนดบทบาทและการตอบสนองของ Agent
        prefix = f"""
        You are a Python expert specializing in data processing and analysis. 
        You are working with a DataFrame. Columns are: {', '.join(df.columns)}.
        Your role is to analyze and manipulate DataFrames in Python.
        Your output must strictly follow this JSON format: {self.output_parser.get_format_instructions()}
        """

        # Suffix: รายละเอียดเฉพาะของบริบทและข้อกำหนดเพิ่มเติม
        suffix = """
        Focus on generating concise and efficient Python code.
        Important Rules:
        1. DO NOT include DataFrame loading code like 'pd.read_csv()' or 'df = pd.read_csv()' - the DataFrame is already loaded as 'df'.
        2. Always work with the existing 'df' variable directly.
        3. Your responses must follow this JSON format strictly:
        {{"query": "description of what the code does",
        "explanation": "detailed explanation of the analysis",
        "code": "print('example')"  // Use single line breaks with \\n, NO triple quotes}}
        4. The code should:
       - Use clear variable names
       - Include comments for complex logic
       - Follow PEP 8 standards
       - Be concise and efficient
        5. Code formatting requirements:
       - Use '\\n' for line breaks (NOT triple quotes)
       - Escape special characters properly
       - NO triple quotes in the code
       - Use single quotes for strings
        6. DO NOT include any text outside of the JSON structure
        """
        return create_pandas_dataframe_agent(
            llm=self.llm,
            df=df,
            agent_type=AgentType.OPENAI_FUNCTIONS,
            prefix=prefix, 
            suffix=suffix,
            verbose=False,
            allow_dangerous_code=True,  
        )

    # def extract_code_snippet(self, response: str) -> str:
    #     """Extract Python code from agent response."""
    #     match = re.search(r'```(?:python|code)?\n(.*?)\n```', response, re.DOTALL)
    #     return match.group(1).strip() if match else response.strip()

    # def execute_code(self, code: str, context: dict):
    #     """Safely execute Python code."""
    #     try:
    #         exec(code, context)
    #     # except Exception as e:
    #     #     logging.error(f"Error executing code: {e}")
    #     except:
    #         pass


    # def run(self, query: str, dataset_key: str):
    #     """Handle user interactions."""
    #     logging.info("Available datasets: %s", ", ".join(self.handler._data.keys()))
    #     agent = self.create_agent(dataset_key)
    #     response = agent.invoke({"input": query})
    #     try:
    #         # Try to parse the output directly if it's already JSON
    #         if isinstance(response['output'], str):
    #             parsed_output = json.loads(response['output'])
    #         else:
    #             parsed_output = response['output']
            
    #         # Validate against our Pydantic model
    #         validated_output = PlotResponse(**parsed_output)
            
    #         print("\n--- Parsed Output ---")
    #         # Change from .dict() to .model_dump()
    #         print(validated_output.model_dump())
    #         exec(validated_output['code'])

                       
    #         # Separate code snippet and explanation
    #         # code_snippet = self.extract_code_snippet(response["output"])
    #         # explanation = response["output"].replace(f"```python\n{code_snippet}\n```", "").strip()
            
    #         # print("\n--- Explanation ---")
    #         # print(explanation)
    #         # print("\n--- Python Code ---")
    #         # print(code_snippet)
    #         # print("\n--- Executing Code ---")
    #         # context = {"pd": pd, "df": self.handler.get_data(dataset_key)}
    #         # self.execute_code(code_snippet, context)
    #     except Exception as e:
    #         logging.error(f"An error occurred: {e}")

    def extract_code_snippet(self, parsed_output: dict) -> str:
        """Extract Python code from parsed JSON output."""
        try:
            return parsed_output.get('code', '')
        except (AttributeError, KeyError):
            logging.error("No code found in the parsed output")
            return ''

    def execute_code(self, code: str, context: dict):
        """Safely execute Python code."""
        if not code:
            logging.warning("No code to execute")
            return
            
        try:
            # Execute the code
            exec(code, context)

        except Exception as e:
            logging.error(f"Error executing code: {str(e)}")
            print(f"Error: {str(e)}")


    def run(self, query: str, dataset_key: str):
        """Handle user interactions."""
        logging.info("Available datasets: %s", ", ".join(self.handler._data.keys()))
        
        try:
            agent = self.create_agent(dataset_key)
            response = agent.invoke({"input": query})
            
            try:
                # Parse the output
                if isinstance(response['output'], str):
                    parsed_output = json.loads(response['output'])
                else:
                    parsed_output = response['output']
                
                # Validate against Pydantic model
                validated_output = PlotResponse(**parsed_output)
                
                print("\n--- Parsed Output ---")
                print(validated_output.model_dump())
                
                # Extract and execute code
                code = self.extract_code_snippet(validated_output.model_dump())
                if code:
                    print("\n--- Executing Code ---")
                    context = {
                        "pd": pd, 
                        "df": self.handler.get_data(dataset_key),
                        "np": np,   
                        "plt":plt, 
                    }
                    self.execute_code(code, context)
                
            except json.JSONDecodeError as e:
                logging.error(f"Error parsing JSON: {e}")
                new_parser = OutputFixingParser.from_llm(parser=self.output_parser, llm=agent)
                new_parser.parse(parsed_output)
            except Exception as e:
                logging.error(f"Error processing output: {e}")
                
        except Exception as e:
            logging.error(f"An error occurred: {e}")



if __name__ == '__main__':
    load_dotenv()
    file_paths = {
        "Financials": "./Financials.csv",
        "McDonald_s_Reviews": "./McDonald_s_Reviews.csv"
    }

    agent = PandasAgent(
        temperature=0.1,
        base_url="https://api.opentyphoon.ai/v1",
        model_name="typhoon-v1.5x-70b-instruct",
        dataset_paths=file_paths
    )
    agent.run(query= 'show me a deep insight and describe it', 
              dataset_key= 'Financials'
              )
