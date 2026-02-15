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
    parser.add_argument('folder_path', help='Path to the folder to check for duplicates')
    
    args = parser.parse_args()
    
    folder_path = args.folder_path
    
    # Check if folder exists
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist.")
        sys.exit(1)
    
    # Check if folder is actually a directory
    if not os.path.isdir(folder_path):
        print(f"Error: '{folder_path}' is not a directory.")
        sys.exit(1)
    
    print(f"Checking for duplicates in: {folder_path}")
    
    # Run duplicate detection (delete=False)
    # The remove_duplicates function will print duplicates without deleting them
    duplicate_count = remove_duplicates(directory=folder_path, delete=False)
    
    print(f"Detection complete. Found {duplicate_count} duplicates.")


if __name__ == "__main__":
    main()