#!/bin/bash
# Start both API and UI servers for development

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  Claude Researcher - Development Mode${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Check if ports are in use
if lsof -Pi :9090 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo -e "${YELLOW}⚠ Port 9090 is in use${NC}"
    echo -e "  Kill it with: ${BLUE}lsof -ti:9090 | xargs kill -9${NC}"
    exit 1
fi

if lsof -Pi :4004 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo -e "${YELLOW}⚠ Port 4004 is in use${NC}"
    echo -e "  Kill it with: ${BLUE}lsof -ti:4004 | xargs kill -9${NC}"
    exit 1
fi

# Get project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo -e "${GREEN}✓${NC} Starting API server on port 9090..."
cd "$PROJECT_ROOT"
python -m api.server > /tmp/claude_api.log 2>&1 &
API_PID=$!
echo $API_PID > /tmp/claude_api.pid

# Wait for API to start
sleep 3
if ! ps -p $API_PID > /dev/null; then
    echo -e "${RED}✗ Failed to start API server${NC}"
    cat /tmp/claude_api.log
    exit 1
fi

echo -e "${GREEN}✓${NC} Starting Next.js UI on port 4004..."
cd "$PROJECT_ROOT/ui"
npm run dev > /tmp/claude_ui.log 2>&1 &
UI_PID=$!
echo $UI_PID > /tmp/claude_ui.pid

# Wait for UI to start
sleep 5
if ! ps -p $UI_PID > /dev/null; then
    echo -e "${RED}✗ Failed to start UI server${NC}"
    cat /tmp/claude_ui.log
    kill $API_PID 2>/dev/null
    exit 1
fi

echo ""
echo -e "${GREEN}✓✓ All servers started!${NC}"
echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "  ${GREEN}Frontend:${NC}  http://localhost:4004"
echo -e "  ${GREEN}API:${NC}       http://localhost:9090"
echo -e "  ${GREEN}API Docs:${NC}  http://localhost:9090/docs"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "${YELLOW}Logs:${NC}"
echo -e "  API: tail -f /tmp/claude_api.log"
echo -e "  UI:  tail -f /tmp/claude_ui.log"
echo ""
echo -e "${YELLOW}Stop:${NC}"
echo -e "  kill \$(cat /tmp/claude_api.pid) \$(cat /tmp/claude_ui.pid)"
echo ""
echo -e "${GREEN}Press Ctrl+C to stop all servers${NC}"
echo ""

# Keep script running and forward signals
trap "echo ''; echo 'Stopping servers...'; kill $API_PID $UI_PID 2>/dev/null; rm -f /tmp/claude_api.pid /tmp/claude_ui.pid; exit 0" INT TERM

# Wait for both processes
wait
