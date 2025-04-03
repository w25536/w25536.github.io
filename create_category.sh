#!/bin/bash

echo "Please enter the title of your category name: "
read sentence

if [ -z "$sentence" ]; then
  echo "Blog post name cannot be empty."
  exit 1
fi

# Use the user input from 'sentence' instead of '$1'
check_sentence=$(echo "$sentence" | tr '[:upper:]' '[:lower:]' | tr ' ' '-' | tr -d '.:')

category_name="${check_sentence}"
categories_dir="categories"

# Create the directory for the category
mkdir -p "${categories_dir}/${category_name}"

# Define the path for index.html inside the category directory
category_path="${categories_dir}/${category_name}/index.html"

# Create or overwrite the index.html with the desired content
cat << EOF > "$category_path"
---
layout: home
category: ${category_name}
---
EOF

echo "Category '${category_name}' created successfully at '${category_path}'."
