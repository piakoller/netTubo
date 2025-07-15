# data_loader.py
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def load_patient_data(file_path: str) -> pd.DataFrame | None:
    """Loads patient data from the specified Excel file."""
    logger.info(f"Attempting to load patient data from: {file_path}")
    try:
        df = pd.read_excel(file_path)
        if df.empty:
            logger.warning(f"Excel file loaded but is empty (or empty after skipping rows): {file_path}")
            print("Warning: The loaded Excel file is empty or contains no data after skipping header rows.")
            return None
        logger.info(f"Successfully loaded {len(df)} patient records from {file_path}")
        return df
    except FileNotFoundError:
        error_msg = f"Error: Excel file not found at {file_path}"
        print(error_msg)
        logger.error(f"Excel file not found: {file_path}")
        return None
    except Exception as e:
        error_msg = f"Error loading Excel file: {e}"
        print(error_msg)
        logger.error(f"Error loading Excel file: {e}", exc_info=True)
        return None