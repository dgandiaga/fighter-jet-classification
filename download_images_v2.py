import os
import argparse
import logging
from icrawler.builtin import BingImageCrawler
from icrawler import ImageDownloader
from curl_cffi import requests
from requests.adapters import HTTPAdapter
import shutil
import json

import random
import time

# Import utility functions from utils
from utils import get_file_count, get_image_hashes, remove_duplicates

# Configure logging
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('download_images.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 1. Low-level override: This forces EVERY request in the session to use 20s
class TimeoutAdapter(HTTPAdapter):
    def send(self, request, **kwargs):
        kwargs['timeout'] = 20  # Global override
        return super().send(request, **kwargs)

# 2. Custom Downloader with your Stealth Headers
class UltimateStealthDownloader(ImageDownloader):
    def download(self, task, default_ext, timeout=20, max_retry=2, **kwargs):
        # 1. Rotate User-Agents to avoid 403 blocks
        agents = [
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 14_3) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2.1 Safari/605.1.15'
        ]
        
        # 2. Add full 'Accept' suite to stop 406 errors
        task['headers'] = {
            'User-Agent': random.choice(agents),
            'Accept': 'image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Referer': 'https://www.bing.com/',
            'Cache-Control': 'no-cache',
            'DNT': '1' # Do Not Track
        }

        # 3. Random Human Delay: Stop the "Machine Speed" trigger
        time.sleep(random.uniform(1.5, 4.0))
        
        return super(UltimateStealthDownloader, self).download(
            task, default_ext, timeout=timeout, max_retry=max_retry, **kwargs
        )
    
def get_image_counts(aircraft_list):
    """Get the count of images for each aircraft class"""
    
    counts = {}
    for aircraft in aircraft_list:
        output_dir = f'dataset/{aircraft}'
        counts[aircraft] = get_file_count(output_dir)
    
    return counts


def persistent_download(config_data, target_count):
    """Download images for each aircraft class"""
    logger.warning(f"Starting download process for {target_count} images per class")
    max_tries = 2
    
    for aircraft, queries in config_data.items():
        
        output_dir = f'dataset/{aircraft}' # Folder for that aircraft
        # Create the folder (and any parent folders) if they don't exist
        os.makedirs(output_dir, exist_ok=True)

        for query in queries:
            n_queries = 0 # Times this query has run
            n_failed_attempts = 0 # Times this query has run without new results
        
            # Keep looping until we actually have target_count files on disk
            while get_file_count(output_dir) < target_count:
                current_count = get_file_count(output_dir)
                
                logger.warning(f"📊 Progress for {aircraft}: {current_count}/{target_count}. Crawling for more...")
                logger.warning(f"📊 Query {query} for {aircraft}, Attempt number {n_queries+1}")
                # We use an offset to skip images we've likely already seen/failed
                # This prevents the 'already downloaded' skip logic from ending the script
                crawler = BingImageCrawler(storage={'root_dir': output_dir},
                                        downloader_cls=UltimateStealthDownloader,
                                        downloader_threads=8
                                        )
                # Apply the adapter to the crawler's internal session
                adapter = TimeoutAdapter()
                crawler.session.mount("https://", adapter)
                crawler.session.mount("http://", adapter)
                
                # We ask for 1.5x what we need to account for dead links
                crawler.crawl(
                    keyword=query,
                    max_num=target_count,
                    offset=int(target_count * n_queries),
                    file_idx_offset='auto'
                )
                n_queries += 1
                # Remove duplicates after each crawl
                remove_duplicates(output_dir)
                
                # Safety break: if we didn't gain any new images in a loop, stop to avoid infinite loop
                if get_file_count(output_dir) == current_count:
                    logger.warning(f"⚠️ No new images found for query {query} for the {n_failed_attempts+1} try")
                    if n_failed_attempts >= max_tries-1:
                        logger.warning(f"⚠️ No new images found for query {query} over {max_tries} loops. Jumping to next query...")
                        break
                    else:
                        n_failed_attempts += 1
                else:
                    n_failed_attempts = 0
    
    logger.warning("Download process completed.")

def clean_dataset_folder():
    """Clean the dataset folder by removing all contents"""
    dataset_path = "./dataset"
    
    if os.path.exists(dataset_path):
        # Remove all files and folders in dataset
        for filename in os.listdir(dataset_path):
            file_path = os.path.join(dataset_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                logger.error(f"Failed to delete {file_path}. Reason: {e}")
    else:
        # Create the directory if it doesn't exist
        os.makedirs(dataset_path, exist_ok=True)

def main():
    """Main function to download images for all aircraft"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Download images for fighter jet classes')
    parser.add_argument('--count', type=int, default=100,
                       help='Number of images to download per class (default: 100)')
    args = parser.parse_args()
    
    # Clean the dataset folder first
    clean_dataset_folder()

    with open('config/queries.json', 'r') as file:
        # 2. Parse the file into a Python dictionary
        config_data = json.load(file)

    persistent_download(config_data, args.count)
    
    # Print image counts per class
    counts = get_image_counts(config_data.keys())
    logger.warning("Download completed. Image counts per class:")
    for aircraft, count in counts.items():
        logger.warning(f"  {aircraft}: {count} images")

if __name__ == "__main__":
    main()