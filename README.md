Bridge Bot 🤖
Welcome to bridgebot! This is an open-source project dedicated to exploring the game of contract bridge through modern software development and machine learning techniques.

This project is a work in progress and is being developed in the open. The primary goal is to create a strong bridge-playing agent, but also to build tools and libraries that are useful to the wider bridge development community. We believe in collaboration and are excited to share our work, even in its early stages.

🚧 Project Status: Under Construction
Current Status: The project is currently in an early, experimental phase. The core simulation loop is partially functional, but there are known issues with the build system and dependency management, especially concerning the compiled C extension (_fastgame).

The instructions below represent the current best path to getting the project running, but please be aware that you may encounter issues. We welcome contributions, and one of the best ways to help right now is by improving the build process and documentation!

Getting Started
These instructions will guide you through setting up a local copy of the project for development and testing.

Prerequisites
Python 3.8+

Go (for running backend servers with Bazel)

Bazel (the primary build tool for this project)

A C compiler (e.g., gcc on Linux)

Installation
Due to the complex nature of this project (mixing Python, Go, and C), the installation is currently a two-step process. Our goal is to simplify this to a single command in the future.

Step 1: Install the C Extension

The core game logic is accelerated with a C extension that has a dependency on numpy. This module must be built and installed into your environment first.

Bash

# Navigate to the C extension's directory
cd src/bridgebot/bridge/fastgame

# Install the C module. This command uses the local setup.py 
# and pyproject.toml to correctly build the extension.
pip install .
Step 2: Install the Main Project

Once the C extension is available in your environment, you can install the main bridgebot package.

Bash

# Navigate back to the project root
cd ../../../..

# Install the main package in editable mode
pip install -e .
How to Run the Simulation
The best way to verify that the core components are working is to run the backend servers and the main simulation script. This requires four separate terminal windows.

Terminal 1: Start the Replay Buffer Server

Bash

# From the project root, run the replay buffer with Bazel
mkdir -p /tmp/replay_buffer
bazel run //bridgebot/replay_buffer -- --data_directory=/tmp/replay_buffer
Terminal 2: Start the Inference Pipe Server

Bash

# From the project root, run the inference pipe server with Bazel
bazel run //bridgebot/inference_pipe
Terminal 3: Run the Fake Inference Client

Bash

# From the project root, run the fake inference client with Bazel
bazel run //bridgebot/inference_pipe/fakeinfer
Terminal 4: Run the Python Simulation Client

Bash

# From the project root, with your virtual environment active
python -m bridgebot.ncard.simulate
If all three terminals run without errors, you have successfully run the core loop of the project!

License
This project is distributed under the MIT License. See the LICENSE file for more details.

We chose the MIT License because it is simple, permissive, and aligns with the open and collaborative spirit of the bridge-dev community. It allows for broad use and modification while ensuring the original work is acknowledged.







