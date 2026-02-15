#!/usr/bin/env python3
"""
Script to calculate the threshold required to determine if two images are duplicates.
This helps fine-tune the threshold value for duplicate detection.
"""

import os
import sys
from PIL import Image
import imagehash

def calculate_hash_difference(image_path1, image_path2):
    """
    Calculate the difference between two image hashes.
    
    Args:
        image_path1 (str): Path to the first image
        image_path2 (str): Path to the second image
    
    Returns:
        int: The hash difference
    """
    # Check if files exist
    if not os.path.exists(image_path1):
        print(f"Error: First image file does not exist: {image_path1}")
        return None
    
    if not os.path.exists(image_path2):
        print(f"Error: Second image file does not exist: {image_path2}")
        return None
    
    try:
        # Load images
        image1 = Image.open(image_path1)
        image2 = Image.open(image_path2)
        
        # Calculate hashes
        hash1 = imagehash.average_hash(image1)
        hash2 = imagehash.average_hash(image2)
        
        # Calculate difference
        difference = hash1 - hash2
        
        return difference
        
    except Exception as e:
        print(f"Error processing images: {e}")
        return None

def main():
    """
    Main function to process command line arguments and calculate threshold.
    """
    # Check if we have the right number of arguments
    if len(sys.argv) != 3:
        print("Usage: python calculate_duplicate_threshold.py <image1_path> <image2_path>")
        print("Example: python calculate_duplicate_threshold.py image1.jpg image2.jpg")
        return
    
    image_path1 = sys.argv[1]
    image_path2 = sys.argv[2]
    
    # Calculate the hash difference
    difference = calculate_hash_difference(image_path1, image_path2)
    
    if difference is not None:
        print(f"Hash difference between '{image_path1}' and '{image_path2}': {difference}")
        print(f"Based on this difference, a threshold of {difference} or lower would consider these images duplicates.")
        print(f"Note: The current default threshold in your script is 12.")
        print(f"Lower threshold = stricter duplicate detection")
        print(f"Higher threshold = more lenient duplicate detection")
    else:
        print("Failed to calculate hash difference.")

if __name__ == "__main__":
    main()