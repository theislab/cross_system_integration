#!/bin/bash

# Check if the requirements file is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <path_to_requirements.txt>"
  exit 1
fi

REQUIREMENTS_FILE=$1

# Check if the file exists
if [ ! -f "$REQUIREMENTS_FILE" ]; then
  echo "Error: File $REQUIREMENTS_FILE does not exist."
  exit 1
fi

# Create a temporary file to store updated requirements
UPDATED_FILE=$(mktemp)

# Get the list of installed packages with versions
PIP_FREEZE=$(pip freeze)

# Process each line in the requirements file
while read -r PACKAGE; do
  if [[ -z "$PACKAGE" || "$PACKAGE" == \#* ]]; then
    # Keep comments and empty lines unchanged
    echo "$PACKAGE" >> "$UPDATED_FILE"
  else
    # Extract the package version from pip freeze
    VERSION=$(echo "$PIP_FREEZE" | grep "^${PACKAGE}==" || true)
    if [ -n "$VERSION" ]; then
      echo "$VERSION" >> "$UPDATED_FILE"
    else
      echo "# $PACKAGE (not found)" >> "$UPDATED_FILE"
    fi
  fi
done < "$REQUIREMENTS_FILE"

# Overwrite the original file with the updated content
mv "$UPDATED_FILE" "$REQUIREMENTS_FILE"

echo "Updated $REQUIREMENTS_FILE with versions from pip freeze."


