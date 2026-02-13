#!/usr/bin/env python3
"""
Script to download 100 images of 5 different fighter jets
using the bing_image_downloader library.
"""

import os
import time
import logging
import shutil
from bing_image_downloader import downloader

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

def download_images_for_aircraft(aircraft_name, search_term, limit=100):
    """Download images for a specific aircraft"""
    
    # Create subfolder for this aircraft
    aircraft_path = os.path.join("./dataset", aircraft_name)
    os.makedirs(aircraft_path, exist_ok=True)
    
    logger.info(f"Starting download of {limit} images of '{search_term}'")
    
    # Download images with error handling and retry logic
    downloaded_count = 0
    max_retries = 5
    
    while downloaded_count < limit:
        try:
            # Download images
            downloader.download(
                search_term,
                limit=limit - downloaded_count,
                output_dir=aircraft_path,
                adult_filter_off=True,
                force_replace=False,
                filter="photo",
                timeout=10
            )
            
            # Count how many images were actually downloaded
            downloaded_count = len(os.listdir(os.path.join(aircraft_path, search_term)))
            logger.info(f"Downloaded {downloaded_count} images so far...")
            
            # If we haven't reached the target, wait and try again
            if downloaded_count < limit:
                logger.info(f"Only {downloaded_count} images downloaded, retrying...")
                time.sleep(5)  # Wait 5 seconds before retry
                
        except Exception as e:
            logger.error(f"Error during download: {e}")
            logger.info(f"Retrying... ({max_retries} attempts left)")
            max_retries -= 1
            if max_retries <= 0:
                logger.error("Max retries exceeded. Stopping download.")
                break
            time.sleep(10)  # Wait 10 seconds before retry
    
    subfolder = os.path.join(aircraft_path, search_term)
    for file in os.listdir(subfolder):
        shutil.move(os.path.join(subfolder, file), os.path.join(aircraft_path, file))

    # 3. Remove the now-empty subfolder
    os.rmdir(subfolder)
    logger.info(f"Download completed for {aircraft_name}. Total images downloaded: {downloaded_count}")
    return downloaded_count

def main():
    """Main function to download images for all aircraft"""
    
    # Clean the dataset folder first
    logger.info("Cleaning dataset folder...")
    clean_dataset_folder()
    
    # List of aircraft to download images for
    aircraft_list = [
        {"name": "Lockheed_Martin_F-35", "search_term": "Lockheed Martin F-35 fighter jet"},
        {"name": "Chengdu_J-20", "search_term": "Chengdu J-20 fighter jet"},
        {"name": "Eurofighter_Typhoon", "search_term": "Eurofighter Typhoon fighter jet"},
        {"name": "Dassault_Rafale", "search_term": "Dassault Rafale fighter jet"},
        {"name": "Saab_Gripen_E_F", "search_term": "Saab Gripen E/F fighter jet"}
    ]
    
    # Download images for each aircraft
    for aircraft in aircraft_list:
        logger.info(f"Starting download for {aircraft['name']}")
        download_images_for_aircraft(aircraft['name'], aircraft['search_term'], 50)
        logger.info(f"Completed download for {aircraft['name']}")

if __name__ == "__main__":
    main()