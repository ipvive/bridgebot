#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../.." >/dev/null 2>&1 && pwd )"

# Run protoc from the project root
python3 -m grpc_tools.protoc \
  --proto_path="$PROJECT_ROOT" \
  --python_out="$PROJECT_ROOT" \
  --grpc_python_out="$PROJECT_ROOT" \
  "bridgebot/pb/alphabridge.proto"

