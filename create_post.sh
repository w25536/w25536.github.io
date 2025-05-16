#!/bin/bash

# Prompt for post title
echo "Enter post title"
read post_title

# Get current date
today=$(date +%Y-%m-%d)

# Create filename by replacing spaces with hyphens
filename=$(echo "$post_title" | sed 's/[:\?\*\"<>|\/\\]//g' | tr ' ' '-')
filename="$today-$filename.md"

# Create the markdown file with front matter
cat > "_posts/$filename" << EOF
---
title: "$post_title"
last_modified_at: $today
categories:
  - 
tags:
  - 
excerpt: "$post_title"
use_math: true
classes: wide
---


EOF

echo "Created file: _posts/$filename"