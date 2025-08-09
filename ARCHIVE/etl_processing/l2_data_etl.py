"""
Module: l2_data_etl.py
Description: ETL pipeline for processing L2 data from a parquet file and loading the transformed data into the L2 database.
Author: Fresh Trading System Team
Date: 2023-10-XX
"""

import argparse
import logging
import sys
from typing import Any

import pandas as pd

# Optional: If you have a dedicated database module, you could import it here.
# from database import get_db_engine

from sqlalchemy import create_engine


def read_parquet_file(file_path: str) -> pd.DataFrame:
    """
    Reads the L2 data parquet file into a DataFrame.

    Args:
        file_path: The path to the parquet file.

    Returns:
        A DataFrame containing the raw L2 data.

    Raises:
        Exception: If the file cannot be read.
    """
    try:
        df = pd.read_parquet(file_path)
        logging.info('Successfully read parquet file: %s', file_path)
        return df
    except Exception as e:
        logging.error('Error reading parquet file %s: %s', file_path, e)
        raise


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw L2 data by handling missing values and performing basic validations.

    Args:
        df: Raw L2 data DataFrame.

    Returns:
        A cleaned DataFrame with dropped or fixed anomalies.
    """
    try:
        # Example cleaning: drop rows with missing values.
        cleaned_df = df.dropna()
        logging.info('Dropped missing values; remaining records: %d', len(cleaned_df))
        return cleaned_df
    except Exception as e:
        logging.error('Error cleaning data: %s', e)
        raise


def transform_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms the cleaned data to fit the database schema.

    Args:
        df: Cleaned L2 data DataFrame.

    Returns:
        A transformed DataFrame that matches the target database schema.
    """
    try:
        df_transformed = df.copy()
        # Example transformation: rename columns if needed
        if 'bidPrice' in df_transformed.columns:
            df_transformed = df_transformed.rename(columns={'bidPrice': 'bid_price'})
        # Additional transformations can be added here
        logging.info('Data transformed to match database schema.')
        return df_transformed
    except Exception as e:
        logging.error('Error transforming data: %s', e)
        raise


def load_data_to_db(df: pd.DataFrame, table_name: str = 'l2_data', if_exists: str = 'append') -> None:
    """
    Loads the transformed data into the database table.

    Args:
        df: Transformed DataFrame to load.
        table_name: Target table name in the database.
        if_exists: Behavior if the table already exists; default is 'append'.

    Raises:
        Exception: If data loading fails.
    """
    try:
        # For demonstration, using a SQLite engine. In production, retrieve engine from configuration or database module.
        engine = create_engine('sqlite:///trading_bot.db')
        df.to_sql(name=table_name, con=engine, if_exists=if_exists, index=False)
        logging.info("Loaded data into database table '%s'", table_name)
    except Exception as e:
        logging.error('Error loading data to DB: %s', e)
        raise


def etl_pipeline(file_path: str) -> None:
    """
    Orchestrates the ETL pipeline: extract, clean, transform, and load the L2 data.

    Args:
        file_path: The path to the raw L2 parquet file.
    """
    logging.info('Starting ETL Pipeline for L2 data processing')
    df_raw = read_parquet_file(file_path)
    df_clean = clean_data(df_raw)
    df_transformed = transform_data(df_clean)
    load_data_to_db(df_transformed)
    logging.info('ETL Pipeline completed successfully.')


def parse_args() -> Any:
    """
    Parses command-line arguments.

    Returns:
        The parsed arguments with the parquet file path.
    """
    parser = argparse.ArgumentParser(description='ETL Pipeline for L2 data processing')
    parser.add_argument('--file', type=str, required=True, help='Path to the raw L2 data parquet file')
    return parser.parse_args()


def main() -> None:
    """
    Main entry point for the ETL process.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    args = parse_args()
    try:
        etl_pipeline(args.file)
    except Exception as e:
        logging.error('ETL Pipeline failed: %s', e)
        sys.exit(1)


if __name__ == '__main__':
    main() 