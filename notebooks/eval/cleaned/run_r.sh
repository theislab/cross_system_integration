#!/bin/bash

# Check if at least one argument (the program.R file) is provided
if [ "$#" -lt 1 ]; then
  echo "Usage: $0 program.R [arguments...]"
  exit 1
fi

# Extract the program.R file (first argument) and all additional arguments
PROGRAM_FILE="$1"
shift  # Remove the first argument so $@ contains the remaining arguments

# Load the OpenMind8 Apptainer module
# Next line does not work. Find the app by loading module and running `which apptainer`
# module load openmind8/apptainer

# Run the command inside an Apptainer container
/cm/shared/openmind8/apptainer/1.1.7/bin/apptainer exec -B /om2/user/moinfar -B /om2/user/khrovati ~/containers/my_ml_verse Rscript "$PROGRAM_FILE" "$@"
