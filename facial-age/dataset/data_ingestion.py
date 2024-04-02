import zipfile
import gdown
from utils.util import logger


def download_and_extract(url, output_path, extract_dir):
    gdown.download(url, output_path)
    logger.info(f'Downloaded data from {url} to {output_path}')

    with zipfile.ZipFile(output_path) as zip_ref:
        zip_ref.extractall(extract_dir)
    logger.info(f'Successfully extracted {output_path} to {extract_dir}')
