#!/usr/bin/env bash
set -euo pipefail

# Install/repair the NVIDIA Container Toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker runtime
sudo nvidia-ctk runtime configure --runtime=docker

# Ensure graphics capability is always available to containers
sudo sed -i \
  's/^#\?\s*env = .*NVIDIA_DRIVER_CAPABILITIES.*/env = ["NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics"]/g' \
  /etc/nvidia-container-runtime/config.toml

# Restart Docker to pick up changes
sudo systemctl restart docker

echo "NVIDIA runtime configured with compute,utility,graphics."
