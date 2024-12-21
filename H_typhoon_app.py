# -----------------------------------------------------------------------
# this is main
from dotenv import load_dotenv
import os
from langchain.agents import AgentExecutor, create_react_agent
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain import hub
from Z_pandas_tst import PandasAgent
from H_datahandle_app import DataHandler
from langchain_core.tools import Tool
from H_summary_app import SummaryAgent
from langchain.agents import AgentExecutor
from langchain.agents import Tool
from langchain.memory import ConversationBufferMemory
import os
import json
from pydantic import BaseModel, Field
import json
from datetime import datetime
import pytz
# Disable LangSmith tracking
os.environ["LANGCHAIN_TRACING_V2"] = "false"

class PlotResponse(BaseModel):
    query: str = Field(description="Description of what is user query")
    response: str = Field(description="Description of output from TyphoonAgent")
    sub_response: dict = Field(description="Response from tool that TyphoonAgent uses")

    def dict(self, *args, **kwargs):
        """Override dict method to customize the output format"""
        return {
            "query": self.query,
            "response": self.response,
            "sub_response": self.sub_response
        }

class TyphoonAgent:
    def __init__(self, temperature: float, base_url: str, model_name: str, dataset_paths: dict, dataset_key: str):
        self.temperature = temperature
        self.base_url = base_url
        self.model = model_name
        self.dataset_key = dataset_key
        self.api_key = os.getenv("TYPHOON_API_KEY")
        self.llm = self.initialize_llm()
        self.memory = self.initialize_memory()
        self.pandas_agent = PandasAgent(temperature, base_url, model_name, dataset_paths)
        self.summary_agent = SummaryAgent(temperature, base_url, model_name)
        self.tools = self.initialize_tools()
        self.agent = self.create_agent()
        self.agent_executor = self.create_agent_executor()


    def initialize_llm(self) -> ChatOpenAI:
        """Initialize the language model."""
        return ChatOpenAI(
            base_url=self.base_url,
            model=self.model,
            api_key=self.api_key,
            temperature=self.temperature,
        )

    def initialize_memory(self):
        """Set up memory for the conversation."""
        return ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    def clear_memory(self):
        """Clear the conversation memory."""
        print('memory cleared !!!')
        return self.memory.clear()

    def initialize_tools(self):
            """Initialize the tools used by the TyphoonAgent."""
            pandas_tool = Tool(
                name="pandas_agent",
                func=self.query_dataframe,
                description=(
                    "REQUIRED for ANY data analysis, plotting, or DataFrame operations. "
                    "Do not provide direct code - instead, use this tool for all data tasks. "
                    "Input: Describe what analysis or plot you want to create. "
                    "Output: Will provide code and explanation."
                ),
            )

            summary_tool = Tool(
                name="summary_agent",
                func=self.summary_answer,
                description="Use for summarizing results or creating concise explanations.",
            )
            return [pandas_tool, summary_tool]
    
    def summary_answer(self, user_input: str) -> None:
        return self.summary_agent.summarize(user_input)


    def query_dataframe(self, user_input: str) -> dict:
        """
        Delegate the user query to the PandasAgent for processing.
        """
        try:
            result = self.pandas_agent.run_and_return_code(user_input, self.dataset_key)
            return result
        except Exception as e:
            return {
                "error": str(e),
                "query": user_input,
                "explanation": "Error occurred while processing the query"
            }


    def create_agent(self):
        """Create a React agent with dataset-aware prompting and structured response requirements."""
        react_prompt = hub.pull("hwchase17/react-chat")
        
        # Get the current dataset information
        if self.dataset_key not in self.pandas_agent.handler._data:
            raise ValueError(f"Dataset '{self.dataset_key}' not found.")
        
        df = self.pandas_agent.handler.get_data(self.dataset_key)
        
        custom_prefix = f"""You are a Data Analysis Supervisor specializing in DataFrame operations.
        CURRENT DATASET: {self.dataset_key}
        AVAILABLE COLUMNS: {', '.join(df.columns)}

        IMPORTANT RULES:
        1. NEVER provide code directly in your response
        2. ALWAYS use pandas_agent for ANY data analysis, plotting, or DataFrame operations
        3. Your main response should be brief and reference the tool outputs
        4. Work with the DataFrame that has these columns: {', '.join(df.columns)}
        
        Remember: ALL code must come from tools, never in direct response."""

        custom_suffix = f"""
        RESPONSE STRUCTURE REQUIREMENTS:
        1. Main Response Format:
        - Keep responses concise and clear
        - Reference the specific tool outputs
        - Maintain exact query consistency
        - Use proper JSON structure
        - Consider available columns: {', '.join(df.columns)}

        2. Tool Usage Guidelines:
        - Pass the exact original query to tools
        - Do not modify or rephrase user queries
        - Use pandas_agent for ALL data operations
        - Use summary_agent for result explanations

        3. Error Handling:
        - Report any issues in processing
        - Maintain consistent error response format
        - Include explanation of errors
        - Check column availability before operations

        4. Dataset Context:
        - Working with dataset: {self.dataset_key}
        - Available columns: {', '.join(df.columns)}
        - Ensure operations use existing columns
        - Validate data types before operations

        5. Additional Guidelines:
        - Never expose internal implementation details
        - Keep responses focused on data analysis
        - Ensure all code comes from tools
        - Maintain professional tone
        - Consider column data types for operations
        """
        
        react_prompt = react_prompt.partial(
            system_message=custom_prefix + custom_suffix
        )
        
        return create_react_agent(llm=self.llm, tools=self.tools, prompt=react_prompt)

    def create_agent_executor(self):
        """Create the agent executor to handle queries."""
        return AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            max_iterations=int(os.getenv("MAX_ITERATIONS", 3)),
            handle_parsing_errors=True,
            return_intermediate_steps=True
        )
 
    def run(self, user_input: str) -> dict:
        """Run the agent with enforced tool usage and proper response capture."""
        
        # Get the raw response from the agent
        raw_response = self.agent_executor.invoke({"input": user_input})
        
        # Extract the main response
        main_response = raw_response.get('output', '')
        
        # Process intermediate steps
        intermediate_steps = raw_response.get('intermediate_steps', [])
        sub_response = {}
        
        # Debug print to see what's in intermediate_steps
        print("Debug - Intermediate steps:", intermediate_steps)
        
        for step in intermediate_steps:
            if len(step) >= 2:
                tool_name = step[0].tool
                tool_output = step[1]
                
                # Handle different types of tool outputs
                if isinstance(tool_output, dict):
                    sub_response[tool_name] = tool_output
                elif isinstance(tool_output, str):
                    # Try to parse JSON if it looks like JSON
                    if tool_output.strip().startswith('{'):
                        try:
                            sub_response[tool_name] = json.loads(tool_output)
                        except json.JSONDecodeError:
                            # If it's not valid JSON, store as is
                            sub_response[tool_name] = {
                                "response": tool_output,
                                "type": "text_response"
                            }
                    else:
                        # Store non-JSON string responses
                        sub_response[tool_name] = {
                            "response": tool_output,
                            "type": "text_response"
                        }
                else:
                    # Convert other types to string
                    sub_response[tool_name] = {
                        "response": str(tool_output),
                        "type": "converted_response"
                    }

        # Create response dictionary
        response_dict = {
            "query": user_input,
            "response": main_response,
            "sub_response": sub_response,
            "metadata": {
                "timestamp": datetime.now(pytz.UTC).strftime('%Y-%m-%d %H:%M:%S'),
                "user": os.getenv("USER", "Nattahphon"),
                "model": self.model,
                "temperature": self.temperature,
                "tools_used": list(sub_response.keys()), 
            }
        }
        
        return response_dict

# Example usage:
if __name__ == '__main__':
    import pandas as pd
    load_dotenv()
    file_paths = {
        "Financials": "./Financials.csv",
        "McDonald_s_Reviews": "./McDonald_s_Reviews.csv"
    }

    typhoon_agent = TyphoonAgent(
        temperature=0.1,
        base_url="https://api.opentyphoon.ai/v1",
        model_name="typhoon-v1.5x-70b-instruct",
        dataset_paths=file_paths,
        dataset_key="Financials"
    )

# """change datatype column 'date' to datetime

    def execute_code(code: str, context: dict):
            """Safely execute Python code."""
            try:
                exec(code, context)
            except:
                pass
    
    handler = DataHandler(dataset_paths=file_paths)
    handler.load_data()
    handler.preprocess_data()
    df = handler.get_data('Financials')

    while True:
            context = {"pd": pd, "df": df}
            user_input = input("Enter your query: ")
            if user_input.lower() == "stop agent":
                typhoon_agent.clear_memory()
                print("Exiting Typhoon Agent...")
                break
            result = typhoon_agent.run(user_input)
            try:
                code = (result['sub_response']['pandas_agent']['code'])
                print('-------------------------------')
                print(json.dumps(result, indent=2))
                print('-------------------------------')
                execute_code(code=code, 
                        context=context)
            except:
                code = (result['response'])
                print('-------------------------------')
                print(json.dumps(result, indent=2))
                print('-------------------------------')
                print(code)