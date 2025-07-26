#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Define the source and output directory.
# This assumes the script is run from the root of the 'bridgebot' project.
PROTOS_DIR=./pb
PYTHON_OUT_DIR=./pb

# Ensure the output directory exists.
mkdir -p ${PYTHON_OUT_DIR}

echo "Generating Python and gRPC code..."

# Generate Python code for all .proto files
# This uses the grpcio-tools package which includes protoc and the necessary plugins.
python3 -m grpc_tools.protoc \
    -I${PROTOS_DIR} \
    --python_out=${PYTHON_OUT_DIR} \
    --grpc_python_out=${PYTHON_OUT_DIR} \
    ${PROTOS_DIR}/alphabridge.proto

echo "Protobuf generation complete."
