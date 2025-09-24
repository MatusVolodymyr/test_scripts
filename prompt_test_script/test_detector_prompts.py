#!/usr/bin/env python3
"""
AI Detector Prompt-Based Testing Script

Generates AI texts from prompts using various models (OpenAI, Anthropic, etc.) 
and tests them against the AI detector API along with human reference texts.

Usage:
    python test_detector_prompts.py --input sample_prompts.csv --generate-and-test
    python test_detector_prompts.py --input sample_prompts.csv --generate-only
    python test_detector_prompts.py --input generated_test_data.csv --test-only

CSV Format:
    type,content,model,temperature,max_tokens,style,domain,expected_label,confidence,language,notes,human_style
"""

import argparse
import csv
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from tqdm import tqdm
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv(".env")

class AITextGenerator:
    """Handles text generation from various AI models."""
    
    def __init__(self):
        self.openai_client = None
        self.anthropic_client = None
        self.costs = {"total": 0.0, "openai": 0.0, "anthropic": 0.0}
        
        # System prompts for different content types to make AI text more human-like
        self.system_prompts = {
            "student_essay": "You are a college student writing an essay for class. Write in a natural, slightly informal academic style with some minor imperfections, personal opinions, and the occasional awkward phrasing that real students use. Don't be overly polished or professional.",
            
            "social_media": "You are writing a social media post or comment. Use casual language, some slang, maybe typos or abbreviations. Be conversational and authentic like a real person posting online. Don't be too formal or structured.",
            
            "blog_post": "You are a blogger writing a personal blog post. Use a conversational tone, share personal thoughts and experiences, and write like you're talking to a friend. Include some tangents and personal anecdotes.",
            
            "forum_comment": "You are commenting on an online forum or discussion board. Write like a real person participating in a discussion - casual, opinionated, maybe a bit rambling. Include personal experiences and observations.",
            
            "email_casual": "You are writing a casual email to a friend or colleague. Use a conversational tone, natural language, and the kind of informal structure people actually use in emails.",
            
            "homework_answer": "You are a student answering a homework question or assignment. Write in a somewhat formal but not perfect academic style, showing your thought process and explaining things as a student would.",
            
            "review_comment": "You are writing a review or comment about a product, service, or experience. Write from personal experience with genuine opinions, both positive and negative aspects.",
            
            "news_comment": "You are commenting on a news article or current event. Express your personal opinion in a conversational way, like a real person discussing the news with others.",
            
            "tutorial_informal": "You are explaining something you know well to someone else in an informal, helpful way. Use simple language and personal examples, like you're helping a friend understand something.",
            
            "personal_story": "You are sharing a personal experience or story. Write naturally and conversationally, with the kind of details and tangents that people include when telling real stories."
        }
        
        # Initialize OpenAI client
        openai_key = os.getenv('OPENAI_API_KEY')
        if openai_key:
            try:
                from openai import OpenAI
                self.openai_client = OpenAI(api_key=openai_key)
                print("✅ OpenAI client initialized")
            except ImportError:
                print("⚠️ OpenAI library not installed. Install with: pip install openai")
            except Exception as e:
                print(f"⚠️ OpenAI client initialization failed: {e}")
        
        # Initialize Anthropic client
        anthropic_key = os.getenv('ANTHROPIC_API_KEY')
        if anthropic_key:
            try:
                import anthropic
                self.anthropic_client = anthropic.Anthropic(api_key=anthropic_key)
                print("✅ Anthropic client initialized")
            except ImportError:
                print("⚠️ Anthropic library not installed. Install with: pip install anthropic")
            except Exception as e:
                print(f"⚠️ Anthropic client initialization failed: {e}")
    
    def get_system_prompt(self, prompt_type: str) -> str:
        """Get system prompt for making AI text more human-like."""
        return self.system_prompts.get(prompt_type, "")
    
    def generate_openai_text(self, prompt: str, model: str, temperature: float, max_tokens: int, system_prompt: str = None) -> Dict[str, Any]:
        """Generate text using OpenAI models."""
        if not self.openai_client:
            return {"success": False, "error": "OpenAI client not initialized"}
        
        try:
            # Prepare messages with optional system prompt
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            # Prepare parameters - some models don't support custom temperature
            params = {
                "model": model,
                "messages": messages,
                "max_completion_tokens": max_tokens
            }
            
            # Only add temperature if it's not the default (1.0) or if model supports it
            # GPT-5 models often only support temperature=1.0
            if temperature != 1.0:
                params["temperature"] = temperature
            
            response = self.openai_client.chat.completions.create(**params)
            
            generated_text = response.choices[0].message.content
            tokens_used = response.usage.total_tokens
            
            return {
                "success": True,
                "text": generated_text.strip(),
                "tokens_used": tokens_used
            }
            
        except Exception as e:
            error_str = str(e)
            
            # If temperature is not supported(gpt-5 models), retry with default temperature
            if "temperature" in error_str and "does not support" in error_str:
                try:
                    print(f"⚠️ Model {model} doesn't support custom temperature, retrying with default...")
                    params_retry = {
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_completion_tokens": max_tokens
                        # No temperature parameter - use model default
                    }
                    
                    response = self.openai_client.chat.completions.create(**params_retry)
                    
                    generated_text = response.choices[0].message.content
                    tokens_used = response.usage.total_tokens
                    
                    return {
                        "success": True,
                        "text": generated_text.strip(),
                        "tokens_used": tokens_used,
                    }
                    
                except Exception as retry_e:
                    return {"success": False, "error": f"Retry failed: {str(retry_e)}"}
            
            return {"success": False, "error": error_str}
    
    def generate_anthropic_text(self, prompt: str, model: str, temperature: float, max_tokens: int, system_prompt: str = None) -> Dict[str, Any]:
        """Generate text using Anthropic models."""
        if not self.anthropic_client:
            return {"success": False, "error": "Anthropic client not initialized"}
        
        try:
            # Prepare parameters with optional system prompt
            params = {
                "model": model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [{"role": "user", "content": prompt}]
            }
            
            if system_prompt:
                params["system"] = system_prompt
            
            response = self.anthropic_client.messages.create(**params)
            
            generated_text = response.content[0].text
            tokens_used = response.usage.input_tokens + response.usage.output_tokens
            
            return {
                "success": True,
                "text": generated_text.strip(),
                "tokens_used": tokens_used,
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def generate_text(self, prompt: str, model: str, temperature: float, max_tokens: int, system_prompt: str = None) -> Dict[str, Any]:
        """Generate text using the appropriate client based on model."""
        model_lower = model.lower()
        
        if "gpt" in model_lower:
            return self.generate_openai_text(prompt, model, temperature, max_tokens, system_prompt)
        elif "claude" in model_lower:
            return self.generate_anthropic_text(prompt, model, temperature, max_tokens, system_prompt)
        else:
            return {"success": False, "error": f"Unsupported model: {model}"}


class DetectorAPITester:
    """Tests generated texts against the AI detector API."""
    
    def __init__(self, api_url: str, token: str = None):
        self.api_url = api_url.rstrip('/')
        self.token = token
        self.headers = {}
        
        if token:
            self.headers['Authorization'] = f'Bearer {token}'
        
        self.results = []
        self.stats = {
            'total_samples': 0,
            'successful_predictions': 0,
            'failed_requests': 0,
            'true_positives': 0,
            'true_negatives': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'total_response_time': 0.0
        }
    
    def call_detector_api(self, text: str, detailed: bool = False) -> Dict[str, Any]:
        """Call the detector API for a single text."""
        url = f"{self.api_url}/api/v1/predict"
        
        payload = {
            'text': text,
            'detailed_response': detailed
        }
        
        start_time = time.time()
        
        try:
            response = requests.post(
                url,
                json=payload,
                headers=self.headers,
                timeout=30
            )
            
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                result['response_time_ms'] = response_time * 1000
                result['success'] = True
                return result
            else:
                return {
                    'success': False,
                    'error': f"HTTP {response.status_code}: {response.text}",
                    'response_time_ms': response_time * 1000
                }
                
        except Exception as e:
            response_time = time.time() - start_time
            return {
                'success': False,
                'error': str(e),
                'response_time_ms': response_time * 1000
            }
    
    def test_samples(self, test_data: List[Dict[str, Any]], detailed: bool = False) -> None:
        """Test all samples against the detector API."""
        self.stats['total_samples'] = len(test_data)
        
        print(f"\nTesting {len(test_data)} samples against detector API...")
        print(f"API URL: {self.api_url}")
        print("-" * 60)
        
        for i, sample in enumerate(tqdm(test_data, desc="Testing samples")):
            result = self.call_detector_api(sample['text'], detailed)
            
            # Track response time
            self.stats['total_response_time'] += result.get('response_time_ms', 0)
            
            if result.get('success'):
                self.stats['successful_predictions'] += 1
                
                # Calculate confusion matrix
                predicted_ai = result.get('is_ai', False)
                expected_ai = sample['expected_label'].lower() == 'ai'
                
                if predicted_ai and expected_ai:
                    self.stats['true_positives'] += 1
                elif not predicted_ai and not expected_ai:
                    self.stats['true_negatives'] += 1
                elif predicted_ai and not expected_ai:
                    self.stats['false_positives'] += 1
                else:
                    self.stats['false_negatives'] += 1
                
                # Store detailed result
                self.results.append({
                    'sample_id': i,
                    'text_preview': sample['text'][:100] + "...",
                    'expected_label': sample['expected_label'],
                    'predicted_ai': predicted_ai,
                    'ai_probability': result.get('ai_probability', 0),
                    'is_humanized': result.get('is_humanized'),
                    'humanizer_probability': result.get('humanizer_probability'),
                    'correct': predicted_ai == expected_ai,
                    'response_time_ms': result.get('response_time_ms', 0),
                    'metadata': sample.get('metadata', {}),
                    'generation_info': sample.get('generation_info', {})
                })
                
            else:
                self.stats['failed_requests'] += 1
                self.results.append({
                    'sample_id': i,
                    'text_preview': sample['text'][:100] + "...",
                    'expected_label': sample['expected_label'],
                    'error': result.get('error', 'Unknown error'),
                    'response_time_ms': result.get('response_time_ms', 0),
                    'metadata': sample.get('metadata', {}),
                    'generation_info': sample.get('generation_info', {})
                })
    
    def calculate_metrics(self) -> Dict[str, float]:
        """Calculate performance metrics."""
        tp = self.stats['true_positives']
        tn = self.stats['true_negatives']
        fp = self.stats['false_positives']
        fn = self.stats['false_negatives']
        
        total_correct = tp + tn
        total_predictions = self.stats['successful_predictions']
        
        metrics = {}
        
        if total_predictions > 0:
            metrics['accuracy'] = total_correct / total_predictions
            metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
            metrics['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            p, r = metrics['precision'], metrics['recall']
            metrics['f1_score'] = 2 * (p * r) / (p + r) if (p + r) > 0 else 0
            metrics['avg_response_time_ms'] = self.stats['total_response_time'] / total_predictions
            
        return metrics
    
    def print_summary(self) -> None:
        """Print test summary to console."""
        metrics = self.calculate_metrics()
        
        print("\n" + "="*60)
        print("DETECTOR TEST RESULTS")
        print("="*60)
        print(f"Total samples: {self.stats['total_samples']}")
        print(f"Successful predictions: {self.stats['successful_predictions']}")
        print(f"Failed requests: {self.stats['failed_requests']}")
        print()
        
        if self.stats['successful_predictions'] > 0:
            print("PERFORMANCE METRICS:")
            print(f"Accuracy:  {metrics.get('accuracy', 0):.3f}")
            print(f"Precision: {metrics.get('precision', 0):.3f}")
            print(f"Recall:    {metrics.get('recall', 0):.3f}")
            print(f"F1 Score:  {metrics.get('f1_score', 0):.3f}")
            print()
            print("CONFUSION MATRIX:")
            print(f"True Positives:  {self.stats['true_positives']}")
            print(f"True Negatives:  {self.stats['true_negatives']}")
            print(f"False Positives: {self.stats['false_positives']}")
            print(f"False Negatives: {self.stats['false_negatives']}")
            print()
            print(f"Average response time: {metrics.get('avg_response_time_ms', 0):.1f}ms")
        
        print("="*60)


class PromptBasedTester:
    """Main class that orchestrates generation and testing."""
    
    def __init__(self, api_url: str = None, token: str = None):
        self.generator = AITextGenerator()
        self.tester = DetectorAPITester(
            api_url or os.getenv('DETECTOR_API_URL', 'http://localhost:8000'),
            token or os.getenv('DETECTOR_API_TOKEN')
        )
        self.generated_data = []
    
    def load_prompts(self, csv_file: str) -> List[Dict[str, Any]]:
        """Load prompts and human texts from CSV file."""
        prompts_data = []
        
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                prompts_data.append({
                    'type': row['type'].strip(),
                    'content': row['content'],
                    'model': row.get('model', '').strip(),
                    'temperature': float(row.get('temperature', 0.7)) if row.get('temperature') and row.get('temperature') != 'N/A' else 0.7,
                    'max_tokens': int(row.get('max_tokens', 300)) if row.get('max_tokens') and row.get('max_tokens') != 'N/A' else 300,
                    'style': row.get('style', ''),
                    'domain': row.get('domain', ''),
                    'expected_label': row.get('expected_label', '').strip(),
                    'confidence': float(row.get('confidence', 1.0)) if row.get('confidence') else 1.0,
                    'language': row.get('language', 'en'),
                    'notes': row.get('notes', ''),
                    'human_style': row.get('human_style', '').strip()  # New field for system prompt type
                })
        
        print(f"Loaded {len(prompts_data)} items from {csv_file}")
        return prompts_data
    
    def generate_texts(self, prompts_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate AI texts from prompts and include human texts."""
        test_data = []
        
        print("\nGenerating AI texts from prompts...")
        print("-" * 60)
        
        for item in tqdm(prompts_data, desc="Processing items"):
            if item['type'].lower() == 'prompt':
                # Get system prompt for human-like generation
                system_prompt = self.generator.get_system_prompt(item.get('human_style', ''))
                
                # Generate AI text with system prompt
                result = self.generator.generate_text(
                    item['content'],
                    item['model'],
                    item['temperature'],
                    item['max_tokens'],
                    system_prompt
                )
                
                if result['success']:
                    test_data.append({
                        'text': result['text'],
                        'expected_label': item['expected_label'],
                        'metadata': {
                            'type': 'generated',
                            'model': item['model'],
                            'temperature': item['temperature'],
                            'max_tokens': item['max_tokens'],
                            'style': item['style'],
                            'domain': item['domain'],
                            'confidence': item['confidence'],
                            'language': item['language'],
                            'original_prompt': item['content'][:100] + "..."
                        },
                        'generation_info': {
                            'tokens_used': result['tokens_used'],
                            'model': item['model']
                        }
                    })
                else:
                    print(f"⚠️ Generation failed for prompt: {item['content'][:50]}...")
                    print(f"   Error: {result['error']}")
                    
            elif item['type'].lower() == 'human_text':
                # Include human text as-is
                test_data.append({
                    'text': item['content'],
                    'expected_label': item['expected_label'],
                    'metadata': {
                        'type': 'human',
                        'style': item['style'],
                        'domain': item['domain'],
                        'confidence': item['confidence'],
                        'language': item['language'],
                        'notes': item['notes']
                    },
                    'generation_info': {}
                })
        
        print(f"\n✅ Generated {len([d for d in test_data if d['metadata']['type'] == 'generated'])} AI texts")
        print(f"✅ Included {len([d for d in test_data if d['metadata']['type'] == 'human'])} human texts")

        self.generated_data = test_data
        return test_data
    
    def save_generated_data(self, output_file: str) -> None:
        """Save generated test data to CSV file."""
        if not self.generated_data:
            print("No generated data to save")
            return
        
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['text', 'expected_label', 'type', 'model', 'domain', 'style', 'confidence', 'language', 'tokens_used'])
            
            for item in self.generated_data:
                writer.writerow([
                    item['text'],
                    item['expected_label'],
                    item['metadata']['type'],
                    item['metadata'].get('model', 'N/A'),
                    item['metadata']['domain'],
                    item['metadata']['style'],
                    item['metadata']['confidence'],
                    item['metadata']['language'],
                    item.get('generation_info', {}).get('tokens_used', 0),
                ])
        
        print(f"Generated test data saved to: {output_file}")
    
    def generate_report(self, output_file: str) -> None:
        """Generate comprehensive test report."""
        metrics = self.tester.calculate_metrics()
        
        report = {
            'test_summary': {
                'timestamp': datetime.now().isoformat(),
                'total_samples': self.tester.stats['total_samples'],
                'successful_predictions': self.tester.stats['successful_predictions'],
                'failed_requests': self.tester.stats['failed_requests'],
                **metrics
            },
            'confusion_matrix': {
                'true_positives': self.tester.stats['true_positives'],
                'true_negatives': self.tester.stats['true_negatives'],
                'false_positives': self.tester.stats['false_positives'],
                'false_negatives': self.tester.stats['false_negatives']
            },
            'detailed_results': self.tester.results
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"Detailed report saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='AI Detector Prompt-Based Testing')
    parser.add_argument('--input', required=True, help='Input CSV file with prompts and human texts')
    parser.add_argument('--api-url', help='Detector API base URL (or use DETECTOR_API_URL env var)')
    parser.add_argument('--token', help='Detector API authentication token (or use DETECTOR_API_TOKEN env var)')
    parser.add_argument('--output-dir', default='test_results', help='Output directory for results')
    parser.add_argument('--detailed', action='store_true', help='Request detailed sentence-level analysis')
    parser.add_argument('--generate-only', action='store_true', help='Only generate texts, don\'t test')
    parser.add_argument('--test-only', action='store_true', help='Only test (input should be generated test data CSV)')
    parser.add_argument('--generate-and-test', action='store_true', help='Generate texts and test them')
    parser.add_argument('--quiet', action='store_true', help='Minimal console output')
    
    args = parser.parse_args()
    
    # Default to generate-and-test if no mode specified
    if not any([args.generate_only, args.test_only, args.generate_and_test]):
        args.generate_and_test = True
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Initialize tester
    tester = PromptBasedTester(args.api_url, args.token)
    
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if args.generate_only or args.generate_and_test:
            # Load prompts and generate texts
            prompts_data = tester.load_prompts(args.input)
            test_data = tester.generate_texts(prompts_data)
            
            # Save generated data
            generated_file = output_dir / f'generated_test_data_{timestamp}.csv'
            tester.save_generated_data(generated_file)
            
            if args.generate_only:
                print(f"\n✅ Generation complete. Use this file for testing:")
                print(f"   python {__file__} --input {generated_file} --test-only")
                return 0
        
        if args.test_only:
            # Load pre-generated test data
            # For test-only mode, we need to load from a different CSV format
            print("⚠️ Test-only mode expects a CSV with generated test data format")
            print("   Use --generate-and-test for complete workflow")
            return 1
        
        if args.generate_and_test:
            # Test the generated data
            tester.tester.test_samples(test_data, args.detailed)
            
            # Print summary
            if not args.quiet:
                tester.tester.print_summary()
            
            # Generate report
            report_file = output_dir / f'prompt_test_report_{timestamp}.json'
            tester.generate_report(report_file)
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())