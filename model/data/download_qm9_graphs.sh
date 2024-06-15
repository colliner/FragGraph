#!/bin/bash

# Define variables
FILE_ID="1RC6qVf1PyNu1UjEMp4ASsZ_SX9F7C_za"
OUTPUT_ZIP="./gdb9_fragment_embedded_graphs.zip"
OUTPUT_DIR="./gdb9_fragment_embedded_graphs/"

# Install gdown if not already installed
if ! command -v gdown &> /dev/null
then
    echo "gdown could not be found. Installing gdown..."
    pip install gdown
fi

# Download the file using gdown
echo "Downloading the file from Google Drive..."
gdown --id $FILE_ID -O $OUTPUT_ZIP

# Create the output directory if it doesn't exist
echo "Creating the output directory..."
mkdir -p $OUTPUT_DIR

# Unzip the file into the output directory
echo "Unzipping the file..."
unzip -o $OUTPUT_ZIP -d $OUTPUT_DIR

# Confirm the contents
echo "Files in the output directory:"
ls $OUTPUT_DIR

