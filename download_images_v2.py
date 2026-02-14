import os
from icrawler.builtin import BingImageCrawler
import shutil

def get_file_count(directory):
    if not os.path.exists(directory):
        return 0
    return len([f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))])

def persistent_download(aircraft_list, target_count):

    for aircraft in aircraft_list:
        output_dir = f'dataset/{aircraft["name"]}'
        
        # Keep looping until we actually have 100 files on disk
        while get_file_count(output_dir) < target_count:
            current_count = get_file_count(output_dir)
            needed = target_count - current_count
            
            print(f"📊 Progress: {current_count}/{target_count}. Crawling for more...")
            
            # We use an offset to skip images we've likely already seen/failed
            # This prevents the 'already downloaded' skip logic from ending the script
            crawler = BingImageCrawler(storage={'root_dir': output_dir})
            
            # We ask for 1.5x what we need to account for dead links
            crawler.crawl(
                keyword=aircraft['search_term'], 
                max_num=int(needed * 1.5), 
                offset=current_count, 
                file_idx_offset=current_count
            )
            
            # Safety break: if we didn't gain any new images in a loop, stop to avoid infinite loop
            if get_file_count(output_dir) == current_count:
                print("⚠️ No new images found. Try a broader search term.")
                break

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
                print(f"Failed to delete {file_path}. Reason: {e}")
    else:
        # Create the directory if it doesn't exist
        os.makedirs(dataset_path, exist_ok=True)


if __name__ == "__main__":
    """Main function to download images for all aircraft"""
    
    # Clean the dataset folder first
    clean_dataset_folder()

    # List of aircraft to download images for
    aircraft_list = [
        {"name": "Lockheed_Martin_F-35", "search_term": "Lockheed Martin F-35 fighter jet"},
        {"name": "Chengdu_J-20", "search_term": "Chengdu J-20 fighter jet"},
        {"name": "Eurofighter_Typhoon", "search_term": "Eurofighter Typhoon fighter jet"},
        {"name": "Dassault_Rafale", "search_term": "Dassault Rafale fighter jet"},
        {"name": "Saab_Gripen", "search_term": "Saab Gripen fighter jet"}
    ]
    persistent_download(aircraft_list, 100)