#!/bin/bash

# ML Training Platform - Setup Script
# This script sets up the complete environment for running the platform

set -e

echo "============================================"
echo "ML Training Platform - Setup"
echo "============================================"
echo ""

# Check for required tools
check_command() {
    if ! command -v $1 &> /dev/null; then
        echo "ERROR: $1 is required but not installed."
        return 1
    fi
    return 0
}

echo "Checking dependencies..."
check_command python3 || exit 1
check_command pip3 || check_command pip || exit 1
check_command node || exit 1
check_command npm || exit 1
check_command git || echo "WARNING: git not found, some features may not work"

echo "Python version: $(python3 --version)"
echo "Node.js version: $(node --version)"
echo "All required dependencies found!"
echo ""

# Create storage directories
echo "Creating storage directories..."
mkdir -p backend/storage/{datasets,models,venvs,logs,configs,annotations,exports,pretrained_models,detectx_builds}
mkdir -p backend/database
echo "Storage directories created."
echo ""

# Setup backend
echo "Setting up backend..."
cd backend

if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

cd ..
echo "Backend setup complete!"
echo ""

# Setup frontend
echo "Setting up frontend..."
cd frontend

echo "Installing Node.js dependencies..."
npm install

# Copy .env.example to .env if it doesn't exist
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo "Created .env from .env.example"
    fi
fi

cd ..
echo "Frontend setup complete!"
echo ""

echo "============================================"
echo "Setup Complete!"
echo "============================================"
echo ""
echo "To start the platform:"
echo ""
echo "  Option 1 - Using start script:"
echo "    ./start.sh"
echo ""
echo "  Option 2 - Manual (Development):"
echo "    Terminal 1: cd backend && source venv/bin/activate && python main.py"
echo "    Terminal 2: cd frontend && npm run dev"
echo ""
echo "  Option 3 - Docker:"
echo "    docker-compose up --build"
echo ""
echo "  Option 4 - Production (Single Server):"
echo "    cd frontend && npm run build && cd .."
echo "    cd backend && source venv/bin/activate && python main.py"
echo ""
echo "Access the platform at:"
echo "  - Frontend: http://localhost:3000 (dev) or http://localhost:8000 (prod)"
echo "  - Backend API: http://localhost:8000"
echo "  - API Docs: http://localhost:8000/docs"
echo ""
