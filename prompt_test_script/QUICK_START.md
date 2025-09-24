# 🚀 Quick Start Guide

This folder contains everything you need to test your AI detector with AI-generated content. **No technical knowledge required!**

## For Windows Users 💻

1. **Double-click** `setup_and_run.bat`
2. Wait for setup to complete
3. Add your API keys when prompted (the .env file will open automatically)
4. **Double-click** `run_test.bat` to run tests

## For Mac/Linux Users 🐧🍎

1. **Double-click** `setup_and_run.sh` (or run `./setup_and_run.sh` in terminal)
2. Wait for setup to complete  
3. Add your API keys when prompted
4. **Double-click** `run_test.sh` (or run `./run_test.sh`) to run tests

## What You Need 🔑

### API Keys (Required for text generation)
- **OpenAI API Key**: Get from [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
- **Anthropic API Key**: Get from [console.anthropic.com](https://console.anthropic.com/)

### Your AI Detector (Required for testing)
- Make sure your AI detector API is running (usually at `http://localhost:8000`)
- The setup script will test the connection for you

## What Happens During Setup ⚙️

1. ✅ Checks if Python is installed
2. ✅ Creates a safe virtual environment 
3. ✅ Installs all required packages
4. ✅ Creates configuration files
5. ✅ Tests your detector API connection
6. ✅ Creates easy-to-use run scripts

## Running Tests 🧪

After setup, you have several options:

### Option 1: Super Easy (Recommended for beginners)
- **Windows**: Double-click `run_test.bat`
- **Mac/Linux**: Double-click `run_test.sh`

### Option 2: Command Line (For more control)
```bash
# Activate the environment first
source venv/bin/activate          # Mac/Linux
# OR
venv\Scripts\activate.bat         # Windows

# Then run tests
python test_detector_prompts.py --input sample_prompts.csv --generate-and-test
```

## What You'll Get 📊

The test will:
- Generate AI texts using GPT and Claude models
- Test them against your detector
- Show you accuracy, precision, recall, and F1 scores
- Save detailed reports in `test_results/` folder
- Show estimated API costs

Example output:
```
DETECTOR TEST RESULTS
====================================
Total samples: 25
Successful predictions: 25

PERFORMANCE METRICS:
Accuracy:  0.920
Precision: 0.900
Recall:    0.947
F1 Score:  0.923

💰 Total estimated cost: $0.0234
```

## Troubleshooting 🔧

### "Python not found"
- Install Python from [python.org/downloads](https://www.python.org/downloads/)
- Make sure to check "Add Python to PATH" during installation

### "API key not working"
- Double-check your API keys in the `.env` file
- Make sure you have credits in your OpenAI/Anthropic accounts
- Remove any extra spaces around the key

### "Cannot reach detector API"
- Make sure your AI detector is running
- Check if it's accessible at `http://localhost:8000/docs`
- Update `DETECTOR_API_URL` in `.env` if using a different address

### "Permission denied" (Mac/Linux)
- Run: `chmod +x setup_and_run.sh run_test.sh`

## Cost Information 💰

API usage costs are typically very low for testing:
- **GPT-3.5-turbo**: ~$0.002 per 1K tokens
- **GPT-4**: ~$0.03 per 1K tokens  
- **Claude-3-haiku**: ~$0.00025 per 1K tokens

A typical test run costs **less than $0.05**.

## Need Help? 🤔

1. **Check the detailed README.md** for more technical information
2. **Look at sample_prompts.csv** to understand the input format
3. **Check test_results/** folder for detailed reports after running tests

---

**That's it! The setup script handles everything else automatically.** 🎉