#!/bin/zsh
git add .
git commit -m "$(date '+%Y-%m-%d') : add info"
git push -f origin main

