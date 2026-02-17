#!/usr/bin/env python3
"""
Script to execute commands listed in config/experiments.json
"""

import json
import subprocess
import sys
import os


def run_experiments():
    """
    Read commands from config/experiments.json and execute them
    """
    # Check if config file exists
    config_path = 'config/experiments.json'
    if not os.path.exists(config_path):
        print(f"Error: Configuration file {config_path} not found!")
        return False
    
    # Read the commands from JSON file
    try:
        with open(config_path, 'r') as f:
            commands = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {config_path}: {e}")
        return False
    except Exception as e:
        print(f"Error reading {config_path}: {e}")
        return False
    
    # Execute each command
    print(f"Executing {len(commands)} commands from {config_path}")
    print("-" * 50)
    
    success_count = 0
    for i, command in enumerate(commands, 1):
        print(f"Command {i}/{len(commands)}:")
        print(f"  {command}")
        print("  Status: RUNNING")
        print("-" * 50)
        
        try:
            # Execute the command with real-time output
            # Using shell=True to allow proper terminal interaction
            result = subprocess.run(
                command,
                shell=True,
                check=True,
                capture_output=False,  # Set to False to allow real-time output
                text=True
            )
            
            print("  Status: SUCCESS")
            success_count += 1
            
        except subprocess.CalledProcessError as e:
            print("  Status: FAILED")
            print(f"  Return code: {e.returncode}")
            print(f"  Command: {e.cmd}")
            if e.stderr:
                print(f"  Error output: {e.stderr}")
            return False
            
        except Exception as e:
            print("  Status: ERROR")
            print(f"  Exception: {e}")
            return False
            
        print("-" * 50)
    
    print(f"Execution completed. {success_count}/{len(commands)} commands succeeded.")
    return success_count == len(commands)


if __name__ == "__main__":
    success = run_experiments()
    sys.exit(0 if success else 1)