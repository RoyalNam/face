import gdown
import zipfile


def download_and_extract(url, output_path, extract_dir):
    gdown.download(url, output_path)

    # Extract the zip file
    with zipfile.ZipFile(output_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
