#!/bin/bash

# Get the current date and time
EXECUTION_DATE=$(date "+%Y%m%d-%H%M")
YEAR=$(date "+%Y")
MONTH=$(date "+%m")

# Define the project directory and logs directory
PROJECT_DIR=$PWD
LOGS_DIR=${PROJECT_DIR}/logs/${YEAR}/${MONTH}

# Create the logs directory if it doesn't exist
mkdir -p ${LOGS_DIR}

# Log start message
echo "================================== Start garbage classification training ====================================="

# Execute the notebook/script using papermill
papermill C:\\Users\\ass_s\\OneDrive\\Documenti\\Garbage_Classification\\notebooks\\garbage_classification.ipynb \
"${LOGS_DIR}/${EXECUTION_DATE}-garbage-classification-artifact.ipynb" \
-k original_venv --report-mode --log-output --no-progress-bar

# Check if the execution was successful
if [ $? != 0 ]; then
  echo "ERROR: failure during training!"
  exit 1
fi

# Log success message
echo "================================ SUCCESS: Done garbage classification training ==================================="
