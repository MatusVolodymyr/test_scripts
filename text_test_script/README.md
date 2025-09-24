# AI Detector API Testing Script

Simple script to test the AI detector API with labeled text data.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage (Local API)
```bash
python test_detector_api.py --input sample_test_data.csv
```

### Test Remote API with Authentication
```bash
python test_detector_api.py \
  --input your_test_data.csv \
  --api-url https://your-api-server.com \
  --token your-auth-token
```

### Get Detailed Analysis
```bash
python test_detector_api.py \
  --input test_data.csv \
  --detailed \
  --output-dir results/
```

## CSV Format

Your test data CSV should have these columns:
```csv
text,label,model_name,word_count,sentence_count,domain,source,confidence,language
```

- **text**: The text to test
- **label**: Expected result ("human" or "ai")
- **model_name**: AI model used (if AI text), or "N/A" for human
- **word_count**: Number of words in text
- **sentence_count**: Number of sentences
- **domain**: Content domain (technology, science, news, etc.)
- **source**: Where text came from (manual, synthetic, social_media, etc.)
- **confidence**: How confident you are in the label (0.0-1.0)
- **language**: Text language (en, es, fr, etc.)

## Output

The script generates:
- Console summary with accuracy, precision, recall, F1-score
- JSON file with detailed results in `test_results/` directory
- Confusion matrix and response time statistics

## Example Output

```
Testing 100 samples...
API URL: http://localhost:8000
Testing samples: 100%|████████████| 100/100 [00:45<00:00,  2.21it/s]

============================================================
TEST RESULTS SUMMARY
============================================================
Total samples: 100
Successful predictions: 98
Failed requests: 2

PERFORMANCE METRICS:
Accuracy:  0.943
Precision: 0.920
Recall:    0.960
F1 Score:  0.940

CONFUSION MATRIX:
True Positives:  48
True Negatives:  46
False Positives: 4
False Negatives: 2

Average response time: 245.3ms
============================================================

Detailed results saved to: test_results/test_results_20250918_143022.json
```