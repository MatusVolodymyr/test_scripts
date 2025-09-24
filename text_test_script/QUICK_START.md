# 🚀 Quick Start Guide - Basic API Testing

This folder contains everything you need to test your AI detector with pre-labeled text data. **No technical knowledge required!**

## For Windows Users 💻

1. **Double-click** `setup_and_run.bat`
2. Wait for setup to complete
3. **Double-click** `run_test.bat` to run tests with your CSV data

## For Mac/Linux Users 🐧🍎

1. **Double-click** `setup_and_run.sh` (or run `./setup_and_run.sh` in terminal)
2. Wait for setup to complete  
3. **Double-click** `run_test.sh` (or run `./run_test.sh`) to run tests

## What You Need 📋

### Your Test Data (Required)
- A CSV file with texts and their labels (AI or human)
- See `sample_test_data.csv` for the expected format
- Columns: `text,label,model_name,word_count,sentence_count,domain,source,confidence,language`

### Your AI Detector (Required)
- Make sure your AI detector API is running (usually at `http://localhost:8000`)
- The setup script will test the connection for you

### No API Keys Needed! 🎉
Unlike the prompt-based testing, this basic API testing doesn't generate new content, so **no OpenAI or Anthropic API keys are required**.

## What Happens During Setup ⚙️

1. ✅ Checks if Python is installed
2. ✅ Creates a safe virtual environment 
3. ✅ Installs required packages (just `requests` and `tqdm`)
4. ✅ Creates configuration files for your API settings
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
python test_detector_api.py --input sample_test_data.csv --detailed
```

## CSV File Format 📄

Your CSV file should have these columns:

| Column | Description | Example |
|--------|-------------|---------|
| `text` | The text to test | `"This is a sample text..."` |
| `label` | Expected result | `human` or `ai` |
| `model_name` | AI model used (if AI text) | `gpt-4`, `claude-3`, or `N/A` |
| `word_count` | Number of words | `245` |
| `sentence_count` | Number of sentences | `12` |
| `domain` | Content area | `technology`, `science`, `news` |
| `source` | Where text came from | `manual`, `social_media`, `news` |
| `confidence` | How sure you are (0-1) | `1.0`, `0.8` |
| `language` | Text language | `en`, `es`, `fr` |

## What You'll Get 📊

The test will:
- Test all your labeled texts against the detector
- Show you accuracy, precision, recall, and F1 scores
- Identify which texts were misclassified
- Save detailed reports in `test_results/` folder
- Show response times and error rates

Example output:
```
DETECTOR TEST RESULTS
====================================
Total samples: 100
Successful predictions: 100

PERFORMANCE METRICS:
Accuracy:  0.920
Precision: 0.900
Recall:    0.947
F1 Score:  0.923

CONFUSION MATRIX:
True Positives:  43
True Negatives:  49
False Positives: 5
False Negatives: 3

Average response time: 142.3ms
```

## Difference from Prompt Testing 🔄

| **Basic API Testing** (this folder) | **Prompt-Based Testing** (other folder) |
|-------------------------------------|------------------------------------------|
| ✅ **Simpler** - no API keys needed | ❗ **Complex** - requires OpenAI/Anthropic keys |
| ✅ **Faster** - tests existing texts | ⏱️ **Slower** - generates then tests |
| ✅ **Cheaper** - no generation costs | 💰 **Costs money** - API usage fees |
| ✅ **Reliable** - no generation failures | ❗ **Can fail** - API limits, network issues |
| ⚠️ **Limited** - only tests provided data | 🎯 **Comprehensive** - tests diverse AI content |

**Use this folder if you:**
- Have pre-labeled text data to test
- Want quick, simple testing
- Don't want to spend money on AI generation
- Are just getting started

**Use the prompt folder if you:**
- Want to test against fresh AI-generated content
- Need comprehensive testing with multiple AI models
- Have API keys and budget for generation
- Want cutting-edge testing capabilities

## Troubleshooting 🔧

### "Python not found"
- Install Python from [python.org/downloads](https://www.python.org/downloads/)
- Make sure to check "Add Python to PATH" during installation

### "Cannot reach detector API"
- Make sure your AI detector is running
- Check if it's accessible at `http://localhost:8000/docs`
- Update `DETECTOR_API_URL` in `.env` if using a different address

### "CSV file not found" or "No sample data"
- Make sure you have a CSV file with your test data
- Check the CSV format matches the expected columns
- Look at `sample_test_data.csv` for an example

### "Permission denied" (Mac/Linux)
- Run: `chmod +x setup_and_run.sh run_test.sh`

## Need Help? 🤔

1. **Check the detailed README.md** for more technical information
2. **Look at sample_test_data.csv** to understand the CSV format
3. **Check test_results/** folder for detailed reports after running tests
4. **Compare with the prompt testing folder** if you need AI content generation

---

**Perfect for quick testing with your existing labeled data!** 🎯