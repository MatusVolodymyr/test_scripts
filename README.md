# AI Detector Testing Suite

A testing suite for AI content detection APIs, featuring both prompt-based content generation and direct API testing capabilities.

## Overview

This project provides two complementary testing approaches for AI detector APIs:

1. **Prompt-Based Testing** (`prompt_test_script/`) - Generate AI content using various models and test detection accuracy
2. **Direct API Testing** (`text_test_script/`) - Test pre-existing labeled text data against detection APIs

## Project Structure

```
test_scripts/
├── README.md                    # This file - main project documentation
├── requirements.txt             # Shared dependencies (if any)
├── .gitignore                   # Git ignore patterns
├── venv/                        # Python virtual environment
├── prompt_test_script/          # Prompt-based AI content generation and testing
│   ├── README.md                # Detailed documentation for prompt testing
│   ├── requirements.txt         # Dependencies for prompt testing
│   ├── test_detector_prompts.py # Main script for prompt-based testing
│   ├── sample_prompts.csv       # Sample prompts for content generation
│   ├── setup_and_run.sh         # Unix setup and run script
│   ├── setup_and_run.bat        # Windows setup and run script
│   ├── run_test.sh              # Quick test runner
│   └── test_results/            # Generated test results and reports
└── text_test_script/            # Direct API testing with labeled data
    ├── README.md                # Detailed documentation for direct testing
    ├── requirements.txt         # Dependencies for direct testing
    ├── test_detector_api.py     # Main script for API testing
    ├── sample_test_data.csv     # Sample labeled test data
    ├── setup_and_run.sh         # Unix setup and run script
    ├── setup_and_run.bat        # Windows setup and run script
    ├── run_test.sh              # Quick test runner
    └── test_results/            # Test results and performance metrics
```

## Quick Start

### Option 1: Prompt-Based Testing
Generate AI content and test detection accuracy:

```bash
cd prompt_test_script/
./setup_and_run.sh
```

### Option 2: Direct API Testing
Test with pre-labeled data:

```bash
cd text_test_script/
./setup_and_run.sh
```

## Features

### Prompt-Based Testing
- **Multi-Model Support**: Generate content using OpenAI GPT models and Anthropic Claude
- **Analytics**: Confusion matrix, precision, recall, F1 score, and response times
- **Flexible Modes**: Generate-only, test-only, or combined workflows

### Direct API Testing
- **Simple Integration**: Test any AI detector API with labeled data
- **Performance Metrics**: Accuracy, precision, recall, and response time analysis
- **Flexible Configuration**: Support for local and remote APIs with authentication
- **Batch Processing**: Efficient testing of large datasets

## Prerequisites

- Python 3.7+
- Virtual environment (recommended)
- API keys for content generation (OpenAI, Anthropic) - for prompt-based testing
- Access to AI detector API endpoint

## Installation

1. Clone or download this repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Choose your testing approach and follow the specific setup instructions in the respective subdirectory

## Configuration

Each testing module has its own configuration requirements:

- **Prompt Testing**: Requires API keys for content generation models
- **Direct Testing**: Requires AI detector API endpoint and authentication details

See the individual README files in each subdirectory for detailed configuration instructions.

## Results and Reports

Both testing modules generate detailed reports including:
- Test accuracy metrics
- Performance statistics
- Confusion matrices (where applicable)
- Response time analysis
- Timestamped results for tracking improvements

Results are saved in the respective `test_results/` directories with timestamps for easy comparison.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Support

For detailed usage instructions and troubleshooting, refer to:
- `prompt_test_script/README.md` for prompt-based testing
- `text_test_script/README.md` for direct API testing
- `prompt_test_script/QUICK_START.md` for quick setup guidance

## License

This project is provided as-is for testing and evaluation purposes.