"""
For larger files, use this code to load data from external storage (s3)
"""

import os
import logging
import pandas as pd
import numpy as np
import boto3
from io import BytesIO

logger = logging.getLogger(__name__)


def load_embeddings_from_s3():
    """Load embeddings file from S3 bucket"""
    try:
        # S3 details
        bucket_name = os.getenv('AWS_S3_BUCKET_NAME')
        file_name = os.getenv('EMBEDDINGS_FILE_NAME', 'biztalk_output.csv')

        # Initialize S3 client
        s3 = boto3.client(
            's3',
            aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        )

        # Download file from S3
        response = s3.get_object(Bucket = bucket_name, Key = file_name)
        file_content = response['Body'].read()

        # Read into pandas
        df = pd.read_csv(BytesIO(file_content))
        df['embedding'] = df['embedding'].apply(eval).apply(np.array)

        logger.info(f"Successfully loaded embeddings from S3: {len(df)} entries")
        return df
    except Exception as e:
        logger.error(f"Error loading embeddings from S3: {str(e)}")
        raise


def load_embeddings_local():
    """Fallback to load embeddings from local file"""
    try:
        file_path = os.getenv('EMBEDDINGS_FILE', 'biztalk_output.csv')
        df = pd.read_csv(file_path)
        df['embedding'] = df['embedding'].apply(eval).apply(np.array)

        logger.info(f"Successfully loaded embeddings from local file: {len(df)} entries")
        return df
    except Exception as e:
        logger.error(f"Error loading embeddings from local file: {str(e)}")
        raise


def load_embeddings():
    """Load embeddings from S3 if configured, otherwise from local file"""
    if os.getenv('AWS_S3_BUCKET_NAME'):
        return load_embeddings_from_s3()
    else:
        return load_embeddings_local()