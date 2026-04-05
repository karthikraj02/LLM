import os
import urllib.request
import zipfile

def download_dataset(dataset="wikitext-2"):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_dir = os.path.join(base_dir, "data", "raw")
    os.makedirs(raw_dir, exist_ok=True)
    
    if dataset == "wikitext-2":
        url = "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip"
        zip_name = "wikitext-2-raw-v1.zip"
    elif dataset == "wikitext-103":
        url = "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip"
        zip_name = "wikitext-103-raw-v1.zip"
    else:
        raise ValueError("Unknown dataset")
        
    zip_path = os.path.join(raw_dir, zip_name)
    
    if not os.path.exists(zip_path):
        print(f"Downloading {url} to {zip_path}...")
        urllib.request.urlretrieve(url, zip_path)
        print("Download complete.")
    else:
        print(f"File {zip_path} already exists. Skipping download.")
        
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(raw_dir)
    print("Done.")

if __name__ == "__main__":
    import sys
    ds = sys.argv[1] if len(sys.argv) > 1 else "wikitext-2"
    download_dataset(ds)
