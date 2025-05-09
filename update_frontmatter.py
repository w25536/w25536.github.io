#!/usr/bin/env python3
import os
import re
import sys
from datetime import datetime

def update_frontmatter(filepath):
    # Read the file
    with open(filepath, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Extract the existing frontmatter
    frontmatter_match = re.search(r'^---\s*\n(.*?)\n---', content, re.DOTALL)
    if not frontmatter_match:
        print(f"No frontmatter found in {filepath}, skipping")
        return False
    
    frontmatter = frontmatter_match.group(1)
    
    # Extract existing fields
    title_match = re.search(r'title:\s*"?(.*?)"?\s*(?:\n|$)', frontmatter)
    title = title_match.group(1) if title_match else os.path.basename(filepath).split('-', 3)[-1].replace('.md', '')
    
    # Extract the date from the filename
    date_match = re.search(r'(\d{4}-\d{2}-\d{2})', os.path.basename(filepath))
    last_modified_at = date_match.group(1) if date_match else datetime.now().strftime('%Y-%m-%d')
    
    # Extract categories but use "skttechacademy" instead
    categories = '  - skttechacademy\n'
    
    # Extract tags
    tags_match = re.search(r'tags:\s*\n((?:  -.*\n)*)', frontmatter)
    if tags_match:
        tags = tags_match.group(1)
    else:
        tags_match = re.search(r'tags:\s*\[(.*?)\]', frontmatter)
        if tags_match:
            tag_list = [tag.strip() for tag in tags_match.group(1).split(',')]
            tags = ''.join([f'  - {tag}\n' for tag in tag_list])
        else:
            # Extract tag from filename or default to empty
            filename = os.path.basename(filepath)
            tag_from_filename = filename.split('-', 3)[-1].replace('.md', '').split('-')[0]
            tags = f'  - {tag_from_filename}\n'
    
    # Extract excerpt, defaulting to title if not found
    excerpt_match = re.search(r'excerpt:\s*"?(.*?)"?\s*(?:\n|$)', frontmatter)
    excerpt = excerpt_match.group(1) if excerpt_match else f"{title} 정리"
    
    # Create new frontmatter
    new_frontmatter = f"""---
title: "{title}"
last_modified_at: {last_modified_at}
categories:
{categories.rstrip()}
tags:
{tags.rstrip()}
excerpt: "{excerpt}"
use_math: true
classes: wide
---"""
    
    # Replace the old frontmatter with the new one
    new_content = content.replace(frontmatter_match.group(0), new_frontmatter)
    
    # Write the updated content back to the file
    with open(filepath, 'w', encoding='utf-8') as file:
        file.write(new_content)
    
    print(f"Updated: {filepath}")
    return True

def main():
    if len(sys.argv) < 2:
        print("Usage: python update_frontmatter.py <file_path> [file_path2 ...]")
        return
    
    for filepath in sys.argv[1:]:
        if os.path.exists(filepath) and filepath.endswith('.md'):
            update_frontmatter(filepath)
        else:
            print(f"File not found or not a markdown file: {filepath}")

if __name__ == "__main__":
    main() 