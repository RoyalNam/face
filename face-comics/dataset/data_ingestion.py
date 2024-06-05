import gdown
import zipfile
from logger import logger


def download_and_extract(url, output_filepath, extract_dir):
    gdown.download(url, output_filepath)
    logger.info(f'Download data from {url} to {output_filepath}')
    with zipfile.ZipFile(output_filepath) as zip_ref:
        zip_ref.extractall(extract_dir)
    logger.info(f'Successfully extracted {output_filepath} to {extract_dir}')
