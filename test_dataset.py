import os
from PIL import Image

def clean_aircraft_data(root_dir):
    bad_files = 0
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(subdir, file)
                try:
                    with Image.open(file_path) as img:
                        # .verify() catches basic corruption
                        img.verify() 
                    
                    # .load() catches "broken data stream" specifically
                    with Image.open(file_path) as img:
                        img.load() 
                        
                except (IOError, OSError, SyntaxError) as e:
                    print(f"Deleting corrupted image: {file_path}")
                    os.remove(file_path)
                    bad_files += 1
                    
    print(f"Cleanup complete. Removed {bad_files} broken images.")

# Run it on your M2 Pro
clean_aircraft_data('dataset')