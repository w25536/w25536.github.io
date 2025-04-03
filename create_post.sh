#!/bin/bash

echo "Please enter the title of your blog post: "
read sentence

# Check if the sentence input is not empty
if [ -z "$sentence" ]; then
  echo "Blog post name cannot be empty."
  exit 1
fi

check_sentence=$(echo "$sentence" | tr '[:upper:]' '[:lower:]' | tr ' ' '-' | tr -d '.:')
# Get the current date in the format YYYY-MM-dd
current_date=$(date +%Y-%m-%d)

# Combine current date and sentence to create the filename
blog_post_name="${current_date}-${check_sentence}"

posts_dir="_posts"
# Full path to the output file inside _posts
blog_post_path="${posts_dir}/${blog_post_name}.md"

# Append the specified content to the file
cat << EOF >> "$blog_post_path"
---
layout: page
title: "${sentence}"
description: ""
headline: ""
tags: [python, 파이썬, torchtext, pytorch, 파이토치, 전처리, data science, 데이터 분석, 딥러닝, 딥러닝 자격증, 머신러닝, 빅데이터]
categories: 
comments: true
published: true
---
EOF

echo "Blog post '${blog_post_name}' created successfully at '${blog_post_path}'."
