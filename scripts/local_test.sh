#!/bin/bash
# Local testing script for conversation analysis Docker images

set -e  # Exit on any error

# Configuration
LOCAL_TAG="local"
REPO_PREFIX="conversation-analysis"

# Get the absolute path to the project root directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Building Docker images for local testing...${NC}"

# Function to build an image
build_image() {
    local agent=$1
    local dockerfile=$2
    local image_name="$REPO_PREFIX-$agent:$LOCAL_TAG"
    
    echo -e "${YELLOW}Building $agent agent image...${NC}"
    docker build -t $image_name -f $PROJECT_ROOT/docker/$dockerfile $PROJECT_ROOT
    
    echo -e "${GREEN}Successfully built $image_name${NC}"
}

# Build Claude agent
build_image "claude" "Dockerfile.claude"

# Build GPT-4 agent
build_image "gpt4" "Dockerfile.gpt4"

echo -e "${GREEN}All images built successfully!${NC}"
echo -e "${YELLOW}Images are available locally:${NC}"
echo -e "  - ${GREEN}$REPO_PREFIX-claude:$LOCAL_TAG${NC}"
echo -e "  - ${GREEN}$REPO_PREFIX-gpt4:$LOCAL_TAG${NC}"
echo -e "${YELLOW}Run with: make run-local${NC}" 