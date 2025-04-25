#!/bin/bash
# Test script for conversation analysis API

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Testing Conversation Analysis API endpoints...${NC}"

# Test GPT-4 agent health endpoint
echo -e "${YELLOW}\nTesting GPT-4 agent health endpoint...${NC}"
response=$(curl -s http://localhost:8001/health)
echo $response
if [[ $response == *"healthy"* ]]; then
  echo -e "${GREEN}✓ GPT-4 agent health check passed!${NC}"
else
  echo -e "${RED}✗ GPT-4 agent health check failed!${NC}"
fi

# Test Claude agent health endpoint
echo -e "${YELLOW}\nTesting Claude agent health endpoint...${NC}"
response=$(curl -s http://localhost:8000/health)
echo $response
if [[ $response == *"healthy"* ]]; then
  echo -e "${GREEN}✓ Claude agent health check passed!${NC}"
else
  echo -e "${RED}✗ Claude agent health check failed!${NC}"
fi

# Test GPT-4 agent with a sample conversation
echo -e "${YELLOW}\nTesting GPT-4 agent with sample conversation...${NC}"
curl -X POST http://localhost:8001/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "message_id": "MessageId(longValue=9)",
        "group_id": "-4673616689",
        "group_title": "NLP Bot Test",
        "message_text": "When is the team meeting scheduled?",
        "message_timestamp": "1745577699",
        "author": {
          "id": "530192978",
          "username": "user1",
          "first_name": "John",
          "last_name": "Doe",
          "is_bot": false
        }
      },
      {
        "message_id": "MessageId(longValue=10)",
        "group_id": "-4673616689",
        "group_title": "NLP Bot Test",
        "message_text": "The team meeting is tomorrow at 2pm",
        "message_timestamp": "1745577750",
        "author": {
          "id": "987654321",
          "username": "user2",
          "first_name": "Jane",
          "last_name": "Smith",
          "is_bot": false
        }
      }
    ],
    "model": "gpt4"
  }'

echo -e "\n"
echo -e "${GREEN}API testing complete!${NC}" 