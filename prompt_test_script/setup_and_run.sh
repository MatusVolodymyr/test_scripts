#!/bin/bash
# AI Detector Prompt Testing - Easy Setup Script
# This script automatically sets up everything needed to run the AI detector testing tool

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo
    echo "=================================================="
    echo "  AI Detector Prompt Testing - Easy Setup"
    echo "=================================================="
    echo
}

# Check if Python is installed
check_python() {
    print_status "Checking Python installation..."
    
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
        PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
        print_success "Found Python $PYTHON_VERSION"
    elif command -v python &> /dev/null; then
        PYTHON_CMD="python"
        PYTHON_VERSION=$(python --version | cut -d' ' -f2)
        print_success "Found Python $PYTHON_VERSION"
    else
        print_error "Python is not installed. Please install Python 3.7 or later."
        print_error "Visit: https://www.python.org/downloads/"
        exit 1
    fi
}

# Set up virtual environment
setup_venv() {
    print_status "Setting up virtual environment..."
    
    if [ -d "venv" ]; then
        print_warning "Virtual environment already exists. Activating existing one..."
    else
        print_status "Creating new virtual environment..."
        $PYTHON_CMD -m venv venv
        print_success "Virtual environment created"
    fi
    
    # Activate virtual environment
    print_status "Activating virtual environment..."
    source venv/bin/activate
    print_success "Virtual environment activated"
    
    # Upgrade pip
    print_status "Upgrading pip..."
    pip install --upgrade pip > /dev/null 2>&1
    print_success "Pip upgraded"
}

# Install dependencies
install_dependencies() {
    print_status "Installing Python dependencies..."
    print_status "This may take a few minutes..."
    
    if pip install -r requirements.txt > /dev/null 2>&1; then
        print_success "All dependencies installed successfully"
    else
        print_error "Failed to install some dependencies"
        print_error "Try running manually: pip install -r requirements.txt"
        exit 1
    fi
}

# Set up .env file
setup_env_file() {
    print_status "Setting up environment configuration..."
    
    if [ -f ".env" ]; then
        print_warning ".env file already exists. Checking configuration..."
        
        # Check if required keys are set
        if grep -q "OPENAI_API_KEY=" .env && grep -q "ANTHROPIC_API_KEY=" .env; then
            print_success "Environment file looks good"
            return
        else
            print_warning "Some API keys may be missing. Please review your .env file."
        fi
    else
        print_status "Creating .env file from template..."
        cp .env.example .env
        print_success ".env file created"
    fi
    
    echo
    echo "=================================================="
    echo "  🔑 API KEY SETUP REQUIRED"
    echo "=================================================="
    echo
    echo "You need to add your API keys to the .env file:"
    echo
    echo "1. OpenAI API Key (for GPT models):"
    echo "   - Get it from: https://platform.openai.com/api-keys"
    echo "   - Add to .env: OPENAI_API_KEY=your-key-here"
    echo
    echo "2. Anthropic API Key (for Claude models):"
    echo "   - Get it from: https://console.anthropic.com/"
    echo "   - Add to .env: ANTHROPIC_API_KEY=your-key-here"
    echo
    echo "3. Your AI Detector API URL (if different from localhost):"
    echo "   - Add to .env: DETECTOR_API_URL=http://your-detector-url:8000"
    echo
    
    read -p "Press Enter when you've added your API keys to the .env file..."
}

# Function to check if .env has API keys
check_api_keys() {
    print_status "Checking API key configuration..."
    
    source .env 2>/dev/null || true
    
    MISSING_KEYS=()
    
    if [ -z "$OPENAI_API_KEY" ] || [ "$OPENAI_API_KEY" = "your-openai-api-key-here" ]; then
        MISSING_KEYS+=("OPENAI_API_KEY")
    fi
    
    if [ -z "$ANTHROPIC_API_KEY" ] || [ "$ANTHROPIC_API_KEY" = "your-anthropic-api-key-here" ]; then
        MISSING_KEYS+=("ANTHROPIC_API_KEY")
    fi
    
    if [ ${#MISSING_KEYS[@]} -eq 0 ]; then
        print_success "API keys configured"
        return 0
    else
        print_warning "Missing or default API keys: ${MISSING_KEYS[*]}"
        return 1
    fi
}

# Test the detector API
test_detector_api() {
    print_status "Testing connection to AI detector API..."
    
    source .env 2>/dev/null || true
    DETECTOR_URL=${DETECTOR_API_URL:-"http://localhost:8000"}
    
    if curl -s -f "$DETECTOR_URL/docs" > /dev/null 2>&1; then
        print_success "AI detector API is accessible at $DETECTOR_URL"
        return 0
    else
        print_warning "Cannot reach AI detector API at $DETECTOR_URL"
        print_warning "Make sure your AI detector is running"
        echo
        echo "To start your detector API, you might need to run:"
        echo "  cd /path/to/your/detector"
        echo "  python -m uvicorn app.main:app --host 0.0.0.0 --port 8000"
        echo
        read -p "Press Enter to continue anyway, or Ctrl+C to exit and start your API..."
        return 1
    fi
}

# Show usage examples
show_usage() {
    echo
    echo "=================================================="
    echo "  🚀 READY TO USE!"
    echo "=================================================="
    echo
    echo "Your testing environment is set up. Here are some commands to try:"
    echo
    echo "1. Basic test with sample data:"
    echo "   ./run_test.sh"
    echo
    echo "2. Generate and test with sample prompts:"
    echo "   python test_detector_prompts.py --input sample_prompts.csv --generate-and-test"
    echo
    echo "3. Generate only (save for later testing):"
    echo "   python test_detector_prompts.py --input sample_prompts.csv --generate-only"
    echo
    echo "4. View detailed help:"
    echo "   python test_detector_prompts.py --help"
    echo
    echo "Results will be saved in the 'test_results/' directory."
    echo
    echo "📚 For more information, see README.md"
    echo
}

# Create a simple run script
create_run_script() {
    print_status "Creating easy run script..."
    
    cat > run_test.sh << 'EOF'
#!/bin/bash
# Simple test runner - activates venv and runs the test

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🚀 Starting AI Detector Test...${NC}"
echo

# Activate virtual environment
source venv/bin/activate

# Run the test
python test_detector_prompts.py --input sample_prompts.csv --generate-and-test

echo
echo -e "${GREEN}✅ Test completed! Check the test_results/ directory for reports.${NC}"
EOF

    chmod +x run_test.sh
    print_success "Created run_test.sh for easy testing"
}

# Main setup function
main() {
    print_header
    
    # Change to script directory
    cd "$(dirname "$0")"
    
    # Check requirements
    check_python
    
    # Set up environment
    setup_venv
    install_dependencies
    setup_env_file
    
    # Check configuration
    if ! check_api_keys; then
        echo
        print_warning "Some API keys are not configured. You can still run the script,"
        print_warning "but text generation will fail for models with missing keys."
        echo
    fi
    
    # Test API connection
    test_detector_api
    
    # Create helper scripts
    create_run_script
    
    # Show usage
    show_usage
    
    print_success "Setup complete! 🎉"
}

# Run main function
main "$@"