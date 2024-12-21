import os
import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# class DataHandler:
#     _instance = None

#     def __new__(cls, *args, **kwargs):
#         if not cls._instance:
#             cls._instance = super(DataHandler, cls).__new__(cls)
#         return cls._instance

#     def __init__(self, dataset_paths=None):
#         """
#         Initialize the DataHandler with dataset paths.
        
#         :param dataset_paths: A dictionary where keys are identifiers (e.g., "df1", "df2")
#                               and values are file paths to the datasets.
#         """
#         if not hasattr(self, '_initialized'):  
#             self._initialized = True
#             if dataset_paths is None:
#                 dataset_paths = {}
#             self.dataset_paths = dataset_paths
#             self._data = {}

#     def load_data(self) -> None:
#         """Load and standardize data from all provided file paths."""
#         if not self.dataset_paths:
#             raise ValueError("No dataset paths provided.")

#         for key, dataset_path in self.dataset_paths.items():
#             if not os.path.exists(dataset_path):
#                 raise FileNotFoundError(f"Dataset file not found at {dataset_path}.")

#             _, ext = os.path.splitext(dataset_path)
#             if ext == ".csv":
#                 try:
#                     self._data[key] = pd.read_csv(dataset_path, encoding="utf-8")
#                 except UnicodeDecodeError:
#                     logging.warning(f"UTF-8 decoding failed for {key}. Trying 'latin1'.")
#                     self._data[key] = pd.read_csv(dataset_path, encoding="latin1")
#             elif ext in [".xls", ".xlsx"]:
#                 self._data[key] = pd.read_excel(dataset_path)
#             else:
#                 raise ValueError(f"Unsupported file extension for {key}: {ext}")

#             # Standardize column names
#             self._data[key].columns = (
#                 self._data[key].columns.str.lower().str.strip().str.replace(" ", "_")
#             )
#             logging.info(f"Data for {key} loaded. Columns: {', '.join(self._data[key].columns)}")

#     def preprocess_data(self) -> None:
#         """Preprocess numeric-like columns for all datasets."""
#         if not self._data:
#             raise ValueError("Data not loaded.")

#         for key, df in self._data.items():
#             numeric_like_cols = [
#                 col
#                 for col in df.columns
#                 if df[col].dtype == "object" and df[col].str.contains(r"[\d,.$€¥-]").any()
#             ]

#             for col in numeric_like_cols:
#                 try:
#                     df[col] = df[col].str.replace(r"[^\d.-]", "", regex=True)
#                     df[col] = pd.to_numeric(df[col], errors="coerce")
#                 except Exception as e:
#                     logging.error(f"Error processing column {col} in dataset {key}: {e}")

#         logging.info("Preprocessing complete.")

#     def get_data(self, key: str) -> pd.DataFrame:
#         """
#         Retrieve the loaded data for a specific key.

#         :param key: The key identifying the dataset (e.g., "df1", "df2").
#         :return: The loaded DataFrame.
#         """
#         if key not in self._data:
#             raise ValueError(f"Data for key '{key}' not loaded.")
#         return self._data[key]


# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class DataHandler:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(DataHandler, cls).__new__(cls)
        return cls._instance

    def __init__(self, dataset_paths=None):
        """
        Initialize the DataHandler with dataset paths.
        :param dataset_paths: A dictionary where keys are identifiers (e.g., "df1", "df2")
                              and values are file paths to the datasets.
        """
        if not hasattr(self, '_initialized'):  
            self._initialized = True
            if dataset_paths is None:
                dataset_paths = {}
            self.dataset_paths = dataset_paths
            self._data = {}

    def load_data(self) -> None:
        """Load and standardize data from all provided file paths."""
        if not self.dataset_paths:
            raise ValueError("No dataset paths provided.")

        for key, dataset_path in self.dataset_paths.items():
            if not os.path.exists(dataset_path):
                raise FileNotFoundError(f"Dataset file not found at {dataset_path}.")

            _, ext = os.path.splitext(dataset_path)
            if ext == ".csv":
                try:
                    self._data[key] = pd.read_csv(dataset_path, encoding="utf-8")
                except UnicodeDecodeError:
                    logging.warning(f"UTF-8 decoding failed for {key}. Trying 'latin1'.")
                    self._data[key] = pd.read_csv(dataset_path, encoding="latin1")
            elif ext in [".xls", ".xlsx"]:
                self._data[key] = pd.read_excel(dataset_path)
            else:
                raise ValueError(f"Unsupported file extension for {key}: {ext}")

            # Standardize column names
            self._data[key].columns = (
                self._data[key].columns.str.lower().str.strip().str.replace(" ", "_")
            )
            logging.info(f"Data for {key} loaded. Columns: {', '.join(self._data[key].columns)}")

    def preprocess_data(self) -> None:
        """Preprocess numeric-like columns for all datasets."""
        if not self._data:
            raise ValueError("Data not loaded.")

        for key, df in self._data.items():
            numeric_like_cols = [
                col
                for col in df.columns
                if df[col].dtype == "object" and df[col].str.contains(r"[\d,.$€¥-]").any()
            ]

            for col in numeric_like_cols:
                try:
                    df[col] = df[col].str.replace(r"[^\d.-]", "", regex=True)
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                except Exception as e:
                    logging.error(f"Error processing column {col} in dataset {key}: {e}")

        logging.info("Preprocessing complete.")

    def get_data(self, key: str) -> pd.DataFrame:
        """
        Retrieve the loaded data for a specific key.
        :param key: The key identifying the dataset (e.g., "df1", "df2").
        :return: The loaded DataFrame.
        """
        if key not in self._data:
            raise ValueError(f"Data for key '{key}' not loaded.")
        return self._data[key]
