#!/usr/bin/env python3
"""
Incremental Results Monitor

This script monitors and displays the progress of ongoing study filtering.
Use this to check results while filtering is still running.
"""

import sys
import json
import time
import os
from pathlib import Path
from datetime import datetime

def find_incremental_files(directory: str = "."):
    """Find incremental progress files in the specified directory."""
    dir_path = Path(directory)
    
    progress_files = list(dir_path.glob("*_INCREMENTAL_progress.txt"))
    kept_files = list(dir_path.glob("*_INCREMENTAL_kept.json"))
    
    return progress_files, kept_files

def display_progress(progress_file: Path):
    """Display the current progress from a progress file."""
    try:
        with open(progress_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print("="*80)
        print(f"PROGRESS UPDATE FROM: {progress_file.name}")
        print("="*80)
        print(content)
        
    except Exception as e:
        print(f"Error reading progress file {progress_file}: {e}")

def display_kept_studies_summary(kept_file: Path, max_studies: int = 10):
    """Display a summary of kept studies from the incremental results."""
    try:
        with open(kept_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print("="*80)
        print(f"KEPT STUDIES SUMMARY FROM: {kept_file.name}")
        print("="*80)
        print(f"Status: {data.get('processing_status', 'Unknown')}")
        print(f"Last Updated: {data.get('last_updated', 'Unknown')}")
        print(f"Progress: {data.get('progress', 'Unknown')}")
        
        incremental = data.get('incremental_results', {})
        print(f"Total Processed: {incremental.get('total_processed', 0)}")
        print(f"Studies Kept: {incremental.get('studies_kept', 0)}")
        print(f"Studies Removed: {incremental.get('studies_removed', 0)}")
        
        studies = data.get('studies', [])
        if studies:
            print(f"\nRecent Kept Studies (showing last {min(max_studies, len(studies))}):")
            print("-" * 60)
            
            for i, study in enumerate(studies[-max_studies:], 1):
                title = study.get('brief_title', study.get('title', 'No title'))
                nct_id = study.get('nct_id', 'Unknown')
                analysis = study.get('publication_analysis', {})
                pubs_found = analysis.get('total_publications_found', 0)
                web_summary = analysis.get('web_results_summary', 'N/A')
                
                print(f"{i:2d}. {title[:60]}")
                print(f"    NCT ID: {nct_id}")
                print(f"    Publications Found: {pubs_found}")
                print(f"    Sources: {web_summary[:80]}...")
                print()
        
    except Exception as e:
        print(f"Error reading kept studies file {kept_file}: {e}")

def monitor_progress(directory: str = ".", watch_mode: bool = False, refresh_interval: int = 30):
    """Monitor filtering progress in the specified directory."""
    
    print(f"Monitoring incremental results in: {Path(directory).absolute()}")
    print(f"Watch mode: {'Enabled' if watch_mode else 'Disabled'}")
    if watch_mode:
        print(f"Refresh interval: {refresh_interval} seconds")
    print("="*80)
    
    while True:
        progress_files, kept_files = find_incremental_files(directory)
        
        if not progress_files and not kept_files:
            print("No incremental filtering files found.")
            print("Make sure you're in the correct directory and filtering is running.")
            if not watch_mode:
                break
            else:
                print(f"Checking again in {refresh_interval} seconds...")
                time.sleep(refresh_interval)
                continue
        
        # Display progress files
        for progress_file in progress_files:
            display_progress(progress_file)
            print()
        
        # Display kept studies summaries
        for kept_file in kept_files:
            display_kept_studies_summary(kept_file)
            print()
        
        if not watch_mode:
            break
        
        print(f"\nRefreshing in {refresh_interval} seconds... (Ctrl+C to stop)")
        try:
            time.sleep(refresh_interval)
            # Clear screen for better readability
            os.system('cls' if os.name == 'nt' else 'clear')
        except KeyboardInterrupt:
            print("\nMonitoring stopped by user.")
            break

def main():
    """Main function for the incremental results monitor."""
    
    if len(sys.argv) == 1:
        # No arguments - show current status
        monitor_progress()
    
    elif len(sys.argv) == 2:
        arg = sys.argv[1]
        
        if arg == "--watch" or arg == "-w":
            # Watch mode - continuously monitor
            monitor_progress(watch_mode=True)
        
        elif arg == "--help" or arg == "-h":
            print("Incremental Results Monitor")
            print("="*50)
            print("Usage:")
            print("  python monitor_incremental.py              # Show current status")
            print("  python monitor_incremental.py --watch      # Continuous monitoring")
            print("  python monitor_incremental.py <directory>  # Monitor specific directory")
            print("  python monitor_incremental.py --help       # Show this help")
            print()
            print("This tool monitors the progress of study filtering and shows")
            print("intermediate results while processing is still running.")
            return
        
        else:
            # Assume it's a directory path
            if Path(arg).exists():
                monitor_progress(directory=arg)
            else:
                print(f"Directory not found: {arg}")
                return
    
    elif len(sys.argv) == 3:
        directory = sys.argv[1]
        if sys.argv[2] == "--watch" or sys.argv[2] == "-w":
            if Path(directory).exists():
                monitor_progress(directory=directory, watch_mode=True)
            else:
                print(f"Directory not found: {directory}")
                return
    
    else:
        print("Too many arguments. Use --help for usage information.")

if __name__ == "__main__":
    main()
