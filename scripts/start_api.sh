#!/bin/bash
# Development script to start the API server

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  Claude Researcher API Server${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# Check if port is in use
PORT="${1:-8080}"
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo -e "${RED}✗ Port $PORT is already in use${NC}"
    echo -e "  Kill it with: ${BLUE}lsof -ti:$PORT | xargs kill -9${NC}"
    exit 1
fi

echo -e "${GREEN}✓${NC} Starting API server on port $PORT..."
echo -e "${GREEN}✓${NC} API Docs: http://localhost:$PORT/docs"
echo -e "${GREEN}✓${NC} Health: http://localhost:$PORT/"
echo -e "${GREEN}✓${NC} Sessions: http://localhost:$PORT/api/sessions/"
echo ""
echo -e "${BLUE}Press Ctrl+C to stop${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Start server
python -m api.server
