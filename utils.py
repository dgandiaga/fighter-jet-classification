import os
import logging
from PIL import Image
import imagehash

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
def get_file_count(directory):
    """Get the number of files in a directory"""
    if not os.path.exists(directory):
        return 0
    return len([f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))])

def get_image_hashes(directory):
    """Get image hashes for all images in a directory"""
    hashes = {}
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            try:
                # Only process image files
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    image = Image.open(file_path)
                    hash_value = imagehash.average_hash(image)
                    hashes[file_path] = hash_value
            except Exception as e:
                logger.warning(f"Failed to process {file_path}: {e}")
    return hashes

def remove_duplicates(directory, threshold=1, delete=True):
    """Remove duplicate images from a directory based on image hashes"""
    logger.info(f"Removing duplicates from {directory}")
    
    # Get all image hashes
    hashes = get_image_hashes(directory)
    
    # Create a list of hash-value pairs to process
    hash_items = list(hashes.items())
    
    # Group similar hashes - we'll use a more robust approach
    unique_hashes = {}
    duplicates = []
    
    # Process each hash
    for i, (file_path, hash_value) in enumerate(hash_items):
        is_duplicate = False
        # Check against all previously identified unique hashes
        for j, (image, existing_hash) in enumerate(unique_hashes.items()):
            # Calculate distance between hashes correctly
            distance = existing_hash - hash_value
            logger.debug(f"Comparing image {file_path} ({hash_value}) with image {image} ({existing_hash}): distance = {distance}")
            if distance <= threshold:
                is_duplicate = True
                logger.info(f"Hash {file_path} is duplicate of hash {image}")
                break
        
        if is_duplicate:
            duplicates.append(file_path)
        else:
            # Add to unique hashes only if not a duplicate
            unique_hashes[file_path] = hash_value
            logger.debug(f"Hash {i} added to unique hashes")
    
    # Remove duplicates (keep only the first occurrence)
    for duplicate in duplicates:
        if delete:
            try:
                os.remove(duplicate)
                logger.info(f"Removed duplicate: {duplicate}")
            except Exception as e:
                logger.error(f"Failed to remove {duplicate}: {e}")
        else:
            logger.warning(f'Duplicate detected: {duplicate}')
    
    logger.warning(f"Detected {len(duplicates)} duplicates from {directory}")
    
    return len(duplicates)