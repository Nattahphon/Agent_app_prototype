import os
import sys
import pandas as pd
import logging
import streamlit as st
from dotenv import load_dotenv
from H_typhoon_app import TyphoonAgent
import matplotlib.pyplot as plt
import io
from H_datahandle_app import DataHandler
from datetime import datetime
import pytz

class StreamlitApp:
    def __init__(self, temperature: float, base_url: str, model_name: str, uploaded_files: list, dataset_key: str, save_directory: str):
        self.temperature = temperature
        self.base_url = base_url
        self.model = model_name
        self.dataset_key = dataset_key
        self.uploaded_files = uploaded_files
        self.save_directory = save_directory
        self.dataset_paths = self.save_uploaded_files()
        self.handler = DataHandler(dataset_paths=self.dataset_paths)
        self.handler.load_data()
        self.handler.preprocess_data()
        self.agent = self.initialize_agent()

        # Initialize session state messages if not already initialized
        if "messages" not in st.session_state:
            st.session_state.messages = []

    def save_uploaded_files(self):
        """Save uploaded files to a specified directory and return a dictionary of file paths."""
        if not os.path.exists(self.save_directory):
            os.makedirs(self.save_directory)

        dataset_paths = {}
        for uploaded_file in self.uploaded_files:
            file_path = os.path.join(self.save_directory, uploaded_file.name)
            try:
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                dataset_paths[uploaded_file.name] = file_path
            except Exception as e:
                st.warning(f"Error saving file {uploaded_file.name}: {e}")
        return dataset_paths

    def initialize_agent(self):
        """Initialize the TyphoonAgent."""
        return TyphoonAgent(
            temperature=self.temperature,
            base_url=self.base_url,
            model_name=self.model,
            dataset_paths=self.dataset_paths,
            dataset_key=self.dataset_key
        )



    def main(self):
        """Main application logic."""
        st.title("Data Analysis Chat Assistant")
        st.write(f"Current Dataset: {self.dataset_key}")

        # Chat input
        user_query = st.chat_input("What would you like to ask?")
        if user_query:
            response = self.agent.run(user_query)
            st.json(response, expanded=2)

# Run the app
if __name__ == "__main__":
    load_dotenv()
    
    st.sidebar.header('Agent Options')
    
    # File upload
    uploaded_files = st.sidebar.file_uploader(
        "Choose CSV files",
        accept_multiple_files=True
    )

    if not uploaded_files:
        st.error("Please upload CSV files to begin analysis.")
        sys.exit(1)

    # Temperature slider
    temp = st.sidebar.select_slider(
        'Set temperature',
        options=[round(i * 0.1, 1) for i in range(0, 11)],
        value=0.1
    )

    # Dataset selector
    dataset_key = st.sidebar.selectbox(
        "Select a dataset",
        [file.name for file in uploaded_files]
    )

    # Initialize and run app
    app = StreamlitApp(
        temperature=temp,
        base_url="https://api.opentyphoon.ai/v1",
        model_name="typhoon-v1.5x-70b-instruct",
        uploaded_files=uploaded_files,
        dataset_key=dataset_key,
        save_directory="./uploaded_files"
    )
    app.main()