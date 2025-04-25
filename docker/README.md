# Docker Quick Reference

## Local Development
```bash
# Build images locally
./scripts/local_test.sh

# Run containers
docker-compose -f compose/docker-compose.local.yml up -d

# Test API
../scripts/test_api.sh
```

## Production Deployment
```bash
# Run from DockerHub images
DOCKER_HUB_USERNAME=yourusername docker-compose -f compose/docker-compose.yml up -d
```

## Publish to DockerHub
```bash
# Manual push
DOCKER_HUB_USERNAME=yourusername VERSION=1.0.0 ./scripts/build_and_push.sh

# Automated via GitHub Actions
# 1. Push to main branch or create tag (v1.0.0)
# 2. GitHub builds and pushes automatically
```

## Configuration
- Set API keys in .env file:
  ```
  ANTHROPIC_API_KEY=your_key_here
  OPENAI_API_KEY=your_key_here
  ``` 