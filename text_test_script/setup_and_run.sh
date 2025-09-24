#!/bin/bash
# AI Detector API Testing - Easy Setup Script
# This script automatically sets up everything needed to run the basic API testing tool

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
    echo "  AI Detector API Testing - Easy Setup"
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
    
    if pip install -r requirements.txt > /dev/null 2>&1; then
        print_success "All dependencies installed successfully"
    else
        print_error "Failed to install some dependencies"
        print_error "Try running manually: pip install -r requirements.txt"
        exit 1
    fi
}

# Create .env file for API configuration
setup_env_file() {
    print_status "Setting up API configuration..."
    
    if [ -f ".env" ]; then
        print_warning ".env file already exists. Checking configuration..."
        source .env 2>/dev/null || true
        if [ -n "$DETECTOR_API_URL" ]; then
            print_success "Environment file looks good"
            return
        fi
    else
        print_status "Creating .env file for API configuration..."
        cat > .env << EOF
# AI Detector API Configuration
DETECTOR_API_URL=http://localhost:8000
DETECTOR_API_TOKEN=

# Optional: Custom output directory
OUTPUT_DIR=test_results
EOF
        print_success ".env file created"
    fi
    
    echo
    echo "=================================================="
    echo "  🔧 API CONFIGURATION"
    echo "=================================================="
    echo
    echo "Default settings created in .env file:"
    echo "- API URL: http://localhost:8000 (local detector)"
    echo "- No authentication token (for local testing)"
    echo
    echo "If your detector API is running elsewhere, update the .env file:"
    echo "- DETECTOR_API_URL=http://your-detector-url:8000"
    echo "- DETECTOR_API_TOKEN=your-auth-token (if needed)"
    echo
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

# Check sample data
check_sample_data() {
    print_status "Checking sample test data..."
    
    if [ -f "sample_test_data.csv" ]; then
        SAMPLE_COUNT=$(tail -n +2 sample_test_data.csv | wc -l)
        print_success "Found sample data with $SAMPLE_COUNT test samples"
    else
        print_warning "No sample_test_data.csv found"
        print_status "You can create your own CSV file with columns: text,label,model_name,word_count,sentence_count,domain,source,confidence,language"
    fi
}

# Create a simple run script
create_run_script() {
    print_status "Creating easy run script..."
    
    cat > run_test.sh << 'EOF'
#!/bin/bash
# Simple test runner - activates venv and runs the basic API test

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}🧪 Starting AI Detector API Test...${NC}"
echo

# Activate virtual environment
source venv/bin/activate

# Load environment variables
source .env 2>/dev/null || true

# Check if sample data exists
if [ ! -f "sample_test_data.csv" ]; then
    echo -e "${YELLOW}⚠️  No sample_test_data.csv found. Please add your test data.${NC}"
    echo "Expected CSV format: text,label,model_name,word_count,sentence_count,domain,source,confidence,language"
    echo
    read -p "Press Enter to continue with a different CSV file, or Ctrl+C to exit..."
    echo
    echo "Available CSV files:"
    ls -1 *.csv 2>/dev/null || echo "No CSV files found in current directory"
    echo
    read -p "Enter CSV filename to test: " CSV_FILE
    if [ ! -f "$CSV_FILE" ]; then
        echo "File not found: $CSV_FILE"
        exit 1
    fi
else
    CSV_FILE="sample_test_data.csv"
fi

# Run the test
echo "Testing with file: $CSV_FILE"
python test_detector_api.py --input "$CSV_FILE" --detailed

echo
echo -e "${GREEN}✅ Test completed! Check the test_results/ directory for detailed reports.${NC}"
EOF

    chmod +x run_test.sh
    print_success "Created run_test.sh for easy testing"
}

# Show usage examples
show_usage() {
    echo
    echo "=================================================="
    echo "  🚀 READY TO USE!"
    echo "=================================================="
    echo
    echo "Your testing environment is set up. Here are some ways to run tests:"
    echo
    echo "1. Easy test with sample data (recommended):"
    echo "   ./run_test.sh"
    echo
    echo "2. Basic test:"
    echo "   python test_detector_api.py --input sample_test_data.csv"
    echo
    echo "3. Detailed test with custom output:"
    echo "   python test_detector_api.py --input your_data.csv --detailed --output-dir results/"
    echo
    echo "4. Test remote API with authentication:"
    echo "   python test_detector_api.py --input data.csv --api-url https://your-api.com --token your-token"
    echo
    echo "5. View help:"
    echo "   python test_detector_api.py --help"
    echo
    echo "Results will be saved in the 'test_results/' directory."
    echo
    echo "📚 For more information, see README.md"
    echo
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
    
    # Check sample data
    check_sample_data
    
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