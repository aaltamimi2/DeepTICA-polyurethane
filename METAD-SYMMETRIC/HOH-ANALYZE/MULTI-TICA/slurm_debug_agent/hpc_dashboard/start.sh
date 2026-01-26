#!/bin/bash
# HPC Dashboard Startup Script

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}╔════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║       HPC Monitor Dashboard            ║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════╝${NC}"
echo ""

# Check for required commands
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: python3 is required${NC}"
    exit 1
fi

# Function to cleanup on exit
cleanup() {
    echo -e "\n${CYAN}Shutting down...${NC}"
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    exit 0
}
trap cleanup SIGINT SIGTERM

# Install backend dependencies if needed
echo -e "${CYAN}[1/3] Checking backend dependencies...${NC}"
cd backend
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi
source venv/bin/activate
pip install -q -r requirements.txt

# Start backend
echo -e "${CYAN}[2/3] Starting backend server...${NC}"
python main.py &
BACKEND_PID=$!
cd ..

# Wait for backend to start
sleep 2

# Check if backend is running
if ! kill -0 $BACKEND_PID 2>/dev/null; then
    echo -e "${RED}Backend failed to start${NC}"
    exit 1
fi

echo -e "${GREEN}Backend running on http://localhost:8080${NC}"

# Start frontend
echo -e "${CYAN}[3/3] Starting frontend...${NC}"
cd frontend

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "Installing frontend dependencies..."
    npm install
fi

# Start React development server
npm start &
FRONTEND_PID=$!
cd ..

echo ""
echo -e "${GREEN}╔════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║   Dashboard ready!                     ║${NC}"
echo -e "${GREEN}║   Frontend: http://localhost:3000      ║${NC}"
echo -e "${GREEN}║   Backend:  http://localhost:8080      ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════╝${NC}"
echo ""
echo "Press Ctrl+C to stop"

# Wait for processes
wait $FRONTEND_PID
