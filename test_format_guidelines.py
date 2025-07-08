#!/usr/bin/env python3
"""
Test script to verify the format_guidelines_for_prompt function output.
This script loads the actual guidelines and NEW_NET_EVIDENCE files and shows
how they are formatted in the prompt.
"""

import sys
from pathlib import Path

# Add the current directory to Python path to import shared_logic
sys.path.append(str(Path(__file__).parent))

from shared_logic import (
    format_guidelines_for_prompt,
    load_structured_guidelines,
    GUIDELINE_SOURCE_DIR,
    NEW_NET_EVIDENCE
)

def test_format_guidelines():
    """Test the format_guidelines_for_prompt function with real data."""
    
    print("="*80)
    print("TESTING format_guidelines_for_prompt FUNCTION")
    print("="*80)
    
    # Load main guidelines
    print(f"\n1. Loading main guidelines from: {GUIDELINE_SOURCE_DIR}")
    structured_guidelines, main_loaded_files = load_structured_guidelines(GUIDELINE_SOURCE_DIR)
    print(f"   Main guidelines loaded: {main_loaded_files}")
    
    # Load additional evidence
    additional_structured = None
    additional_loaded_files = []
    if NEW_NET_EVIDENCE:
        print(f"\n2. Loading additional evidence from: {NEW_NET_EVIDENCE}")
        additional_structured, additional_loaded_files = load_structured_guidelines(Path(NEW_NET_EVIDENCE))
        print(f"   Additional evidence loaded: {len(additional_loaded_files)} files")
        print(f"   First 5 files: {additional_loaded_files[:5]}")
    
    # Format the guidelines
    print(f"\n3. Formatting guidelines for prompt...")
    guidelines_context_string = format_guidelines_for_prompt(
        structured_docs=structured_guidelines,
        additional_structured_docs=additional_structured,
        additional_dir=Path(NEW_NET_EVIDENCE) if NEW_NET_EVIDENCE else None
    )
    
    # Show the structure
    print(f"\n4. FORMATTED PROMPT STRUCTURE:")
    print("-" * 60)
    
    # Show just the tag structure (first 200 chars of each section)
    lines = guidelines_context_string.split('\n')
    in_content = False
    content_lines = 0
    
    for line in lines:
        if line.strip().startswith('<') and line.strip().endswith('>'):
            # This is a tag line
            if in_content:
                print("   [content truncated...]")
                in_content = False
                content_lines = 0
            print(line)
            if not line.strip().startswith('</'):
                in_content = True
        elif in_content:
            content_lines += 1
            if content_lines <= 3:  # Show first 3 lines of content
                print(f"   {line[:100]}{'...' if len(line) > 100 else ''}")
            elif content_lines == 4:
                print("   [... more content ...]")
    
    print("-" * 60)
    
    # Show statistics
    print(f"\n5. STATISTICS:")
    print(f"   Total prompt length: {len(guidelines_context_string):,} characters")
    print(f"   Total lines: {len(lines):,}")
    
    # Count tags
    tag_counts = {}
    for line in lines:
        if line.strip().startswith('<') and line.strip().endswith('>'):
            tag = line.strip()
            if not tag.startswith('</'):
                tag_name = tag[1:-1]  # Remove < and >
                tag_counts[tag_name] = tag_counts.get(tag_name, 0) + 1
    
    print(f"   Tags found:")
    for tag, count in sorted(tag_counts.items()):
        print(f"     - {tag}: {count}")
    
    # Save to file for inspection
    output_file = Path(__file__).parent / "test_prompt_output.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("FORMATTED GUIDELINES FOR PROMPT\n")
        f.write("="*80 + "\n\n")
        f.write(guidelines_context_string)
    
    print(f"\n6. Full output saved to: {output_file}")
    print(f"   You can open this file to see the complete formatted prompt.")
    
    return guidelines_context_string

if __name__ == "__main__":
    try:
        result = test_format_guidelines()
        print(f"\n✅ Test completed successfully!")
        print(f"   Check the output above and the generated test_prompt_output.txt file.")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
