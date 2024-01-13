#!/bin/bash

# Get the current directory name
current_dir=${PWD##*/}

# Check if the current directory is not 'Saprykin_Dmitry'
if [ "$current_dir" != "Saprykin_Dmitry" ]; then
    echo "You are not in the 'Saprykin_Dmitry' directory."
    exit 1
fi

python search/create_search.py