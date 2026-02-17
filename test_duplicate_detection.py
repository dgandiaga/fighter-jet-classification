#!/usr/bin/env python3
"""
Test script for duplicate detection in a folder.
This script uses the duplicate detection function from utils.py
and prints duplicates without deleting them (delete=False).
"""

import os
import sys
import argparse
from utils import remove_duplicates


def main():
    parser = argparse.ArgumentParser(description='Detect duplicates in a folder')
    parser.add_argument('--folder-path', type=str, default='./dataset',
                          help='Path to dataset directory (default: ./dataset)')
    parser.add_argument('--threshold', type=int, default=12,
                          help='Threshold for duplicate detection (default: 12)')
    parser.add_argument('--delete', type=bool, default=False,
                          help='Auto delete duplicates found')
    
    args = parser.parse_args()
    
    folder_path = args.folder_path
    threshold = args.threshold
    delete = args.delete
    
    # Check if folder exists
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist.")
        sys.exit(1)
    
    for folder in os.listdir(folder_path):
        folder_full_path = os.path.join(folder_path, folder)
        # Check if folder is actually a directory
        if not os.path.isdir(folder_full_path):
            print(f"Error: '{folder_full_path}' is not a directory.")
            sys.exit(1)
        
        print(f"Checking for duplicates in: {folder_full_path}")
        
        # Run duplicate detection (delete=False)
        # The remove_duplicates function will print duplicates without deleting them
        duplicate_count = remove_duplicates(directory=folder_full_path, threshold=threshold, delete=delete)
        
        print(f"{folder}: found {duplicate_count} duplicates.")
    for folder in os.listdir(folder_path):
        print(f'{folder}: {len(os.listdir(os.path.join(folder_path, folder)))}')


if __name__ == "__main__":
    main()