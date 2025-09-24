# Prompt-Based AI Detector Testing Script

This directory contains tools for testing the AI detector API using AI-generated content from various models.

## Features

- **Multi-Model Support**: Generate text using OpenAI GPT models and Anthropic Claude
- **Cost Tracking**: Estimates and tracks API usage costs for generation
- **Comprehensive Testing**: Tests both AI-generated and human reference texts
- **Detailed Analytics**: Confusion matrix, precision, recall, F1 score, and response times
- **Flexible Modes**: Generate-only, test-only, or combined workflows

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up API keys by copying `.env.example` to `.env` and filling in your keys:
```bash
cp .env.example .env
# Edit .env with your API keys
```

3. Prepare your detector API:
```bash
# Make sure your detector API is running
# Default: http://localhost:8000
```

## Usage

### Basic Usage (Generate and Test)
```bash
python test_detector_prompts.py --input sample_prompts.csv --generate-and-test
```

### Generate Only (for later testing)
```bash
python test_detector_prompts.py --input sample_prompts.csv --generate-only
```

### Advanced Options
```bash
# Use custom API URL and token
python test_detector_prompts.py \
    --input sample_prompts.csv \
    --api-url http://your-api-url:8000 \
    --token your-auth-token \
    --detailed \
    --output-dir custom_results

# Quiet mode (minimal output)
python test_detector_prompts.py --input sample_prompts.csv --generate-and-test --quiet
```

## Input CSV Format

The input CSV should have these columns:

| Column | Description | Required | Examples |
|--------|-------------|----------|----------|
| `type` | `prompt` or `human_text` | Yes | `prompt`, `human_text` |
| `content` | Prompt text or human text | Yes | `Write a story about...`, `This is human text...` |
| `model` | AI model to use | For prompts | `gpt-4o`, `gpt-3.5-turbo`, `claude-3-5-sonnet` |
| `temperature` | Generation randomness (0-1) | No | `0.7`, `0.9` |
| `max_tokens` | Maximum tokens to generate | No | `300`, `500` |
| `style` | Writing style description | No | `formal`, `casual`, `academic` |
| `domain` | Content domain | No | `technology`, `literature`, `science` |
| `expected_label` | Expected detector result | Yes | `ai`, `human` |
| `confidence` | Confidence in expected label | No | `1.0`, `0.8` |
| `language` | Content language | No | `en`, `es`, `fr` |
| `notes` | Additional notes | No | Any text |
| `human_style` | System prompt type for realistic AI text | For prompts | `student_essay`, `social_media`, `blog_post` |

## Human-Style System Prompts

To make AI-generated text more realistic and human-like, the script uses system prompts based on the `human_style` column:

- **`student_essay`**: College student writing style with minor imperfections and personal opinions
- **`social_media`**: Casual social media posts with slang, abbreviations, and conversational tone  
- **`blog_post`**: Personal blogger style with tangents and anecdotes
- **`forum_comment`**: Online discussion style, casual and opinionated
- **`email_casual`**: Informal email communication with natural language
- **`homework_answer`**: Student answering assignments with academic but imperfect style
- **`review_comment`**: Personal product/service reviews with genuine opinions
- **`news_comment`**: Commenting on current events with personal perspectives
- **`tutorial_informal`**: Explaining concepts in a helpful, friendly way
- **`personal_story`**: Sharing experiences naturally with details and tangents

## Supported AI Models

### OpenAI Models
- `gpt-4`
- `gpt-4-turbo`
- `gpt-3.5-turbo`

### Anthropic Models
- `claude-3-opus-20240229`
- `claude-3-sonnet-20240229`
- `claude-3-haiku-20240307`

## Output Files

The script generates several output files in the `test_results/` directory:

- `generated_test_data_YYYYMMDD_HHMMSS.csv`: Generated texts for later use
- `prompt_test_report_YYYYMMDD_HHMMSS.json`: Comprehensive test results and metrics

## Cost Estimation

The script provides cost estimates for API usage:

```
💰 Total estimated generation cost: $0.0234
💰 OpenAI cost: $0.0120
💰 Anthropic cost: $0.0114
```

**Note**: These are estimates based on public pricing. Actual costs may vary.

## Example Results

```
DETECTOR TEST RESULTS
====================================
Total samples: 25
Successful predictions: 25
Failed requests: 0

PERFORMANCE METRICS:
Accuracy:  0.920
Precision: 0.900
Recall:    0.947
F1 Score:  0.923

CONFUSION MATRIX:
True Positives:  18
True Negatives:  5
False Positives: 1
False Negatives: 1

Average response time: 145.2ms
```

## Environment Variables

Set these in your `.env` file:

- `OPENAI_API_KEY`: Your OpenAI API key
- `ANTHROPIC_API_KEY`: Your Anthropic API key
- `DETECTOR_API_URL`: Your detector API base URL (default: http://localhost:8000)
- `DETECTOR_API_TOKEN`: Authentication token for your detector API (if required)

## Tips

1. **Start Small**: Begin with a few samples to test your setup
2. **Cost Monitoring**: Monitor your API usage, especially with GPT-4
3. **Diverse Prompts**: Use varied prompts to test different content types
4. **Human References**: Include human texts for baseline comparison
5. **Temperature Variation**: Test different temperature values for generation diversity

## Troubleshooting

- **API Key Issues**: Ensure your API keys are valid and have sufficient credits
- **Model Access**: Verify you have access to the requested models
- **Network Issues**: Check your internet connection and API endpoints
- **Rate Limits**: Be aware of API rate limits; the script includes basic error handling