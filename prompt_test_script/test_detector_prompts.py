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
    type,content,model,temperature,max_tokens,style,domain,expected_label,confidence,language,notes
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
        
        # Universal system prompt for generating natural, human-like text
        self.universal_system_prompt_template = """You are writing as a real human with authentic thoughts and natural expression. Your goal is to produce text that sounds genuinely human-written, not AI-generated. This text will later be used to test AI text detectors.

Core guidelines for human-like writing:
- Write with natural flow and authentic voice - imagine you're explaining to a friend
- Include personal touches: "I think," "in my experience," "honestly," "to be fair"
- Use varied sentence structure: mix short punchy statements with longer, meandering thoughts
- Add natural tangents and asides that humans include when they get excited about a topic
- Include some redundancy, minor repetition, or slight contradictions that happen in natural speech
- Use contractions, informal phrases, and conversational connectors ("So," "But here's the thing," "Actually,")
- Don't be overly structured - let ideas flow organically with some messiness
- Include uncertainty, multiple viewpoints, or admissions of not knowing everything
- Add relatable examples from everyday life or common experiences
- Use humor, mild frustration, or genuine enthusiasm where appropriate
- Include some grammatical quirks or colloquialisms that real people use
- Make transitions feel conversational rather than academic ("Speaking of which," "Oh, and another thing")

{length_instruction}

{style_instruction}

{additional_context}

Remember: Write like a knowledgeable person having a genuine conversation. The goal is creating content so authentically human that even sophisticated AI detectors would struggle to identify it as machine-generated. Be engaging, substantial, and real."""
        
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
    
    def get_universal_system_prompt(self, temperature: float = 0.7, max_tokens: int = 500, style: str = "", domain: str = "") -> str:
        """Get the universal system prompt with parameter-based instructions."""
        
        # Length instruction based on max_tokens
        if max_tokens <= 300:
            length_instruction = "Target length: Write 200-400 words - be concise but natural, like a focused conversation."
        elif max_tokens <= 500:
            length_instruction = "Target length: Write 400-600 words - develop your thoughts with examples and personal touches."
        elif max_tokens <= 800:
            length_instruction = "Target length: Write 600-900 words - explore the topic thoroughly with tangents and detailed explanations."
        else:
            length_instruction = "Target length: Write 800+ words - be comprehensive, include multiple perspectives, detailed examples, and natural digressions."
        
        # Style instruction based on temperature
        if temperature <= 0.3:
            style_instruction = """Writing style: Be thoughtful and measured. Include careful reasoning, acknowledge nuances, and show you've considered different angles. Use phrases like "I think," "it seems to me," or "from what I understand." Be precise but still conversational."""
        elif temperature <= 0.7:
            style_instruction = """Writing style: Strike a balanced tone - informative but personable. Mix analytical thinking with relatable examples. Show both expertise and humility ("I've found that..." or "In my experience..."). Be engaging without being overly casual."""
        else:
            style_instruction = """Writing style: Be expressive and enthusiastic! Use vivid language, show excitement about interesting points, include humor or mild exaggeration where natural. Don't be afraid of tangents, personal anecdotes, or showing strong opinions. Sound like someone genuinely passionate about the topic."""
        
        # Additional context based on domain/style
        additional_context = ""
        if domain or style:
            additional_context = f"Context: This is {style} content about {domain}. " if domain and style else f"Context: This is {style or domain} content. "
            additional_context += "Adapt your voice accordingly while maintaining the human authenticity described above."
        
        return self.universal_system_prompt_template.format(
            length_instruction=length_instruction,
            style_instruction=style_instruction,
            additional_context=additional_context
        )

    def _model_restrictions(self, model: str) -> Dict[str, Any]:
        """Return model-specific parameter restrictions."""
        if not model:
            return {}
        model_lower = model.lower()

        # Preset GPT-5 variants ship with fixed decoding settings.
        if model_lower.startswith("gpt-5"):
            return {
                "disallow_temperature": True,
                "disallow_top_p": True,
                "disallow_max_output_tokens": True,
            }

        return {}

    def _call_openai_responses(self, prompt: str, model: str, temperature: float, max_tokens: int, system_prompt: Optional[str]) -> Dict[str, Any]:
        """Call the OpenAI Responses API used by GPT-5 / GPT-4.1 families."""
        # Build inputs in the rich content format expected by the Responses API.
        input_messages: List[Dict[str, Any]] = []
        if system_prompt:
            input_messages.append({
                "role": "system",
                "content": system_prompt
            })
        input_messages.append({
            "role": "user",
            "content": prompt
        })

        params: Dict[str, Any] = {
            "model": model,
            "input": input_messages,
        }

        restrictions = self._model_restrictions(model)
        temperature_allowed = not restrictions.get("disallow_temperature", False)
        max_tokens_allowed = not restrictions.get("disallow_max_output_tokens", False)

        if temperature_allowed and temperature != 1.0:
            params["temperature"] = temperature
        elif not temperature_allowed and temperature != 1.0:
            print(f"ℹ️ Model {model} enforces a fixed temperature; ignoring override {temperature}.")

        if max_tokens_allowed and max_tokens:
            params["max_output_tokens"] = max_tokens
        elif not max_tokens_allowed and max_tokens:
            print(f"ℹ️ Model {model} ignores max_output_tokens; using provider defaults instead.")

        response = self.openai_client.responses.create(**params)

        generated_text = getattr(response, "output_text", None)

        if not generated_text:
            # Fall back to manually assembling text from the structured output.
            aggregated_chunks: List[str] = []
            output_items = getattr(response, "output", None) or []
            for item in output_items:
                item_type = getattr(item, "type", None) or item.get("type")
                if item_type != "message":
                    continue
                content_blocks = getattr(item, "content", None) or item.get("content", [])
                for block in content_blocks:
                    block_type = getattr(block, "type", None) or block.get("type")
                    if block_type in ("text", "output_text"):
                        text_value = getattr(block, "text", None) or block.get("text")
                        if text_value:
                            aggregated_chunks.append(text_value)
            generated_text = "".join(aggregated_chunks)

        if not generated_text:
            return {"success": False, "error": f"Model {model} returned empty content"}

        usage = getattr(response, "usage", None)
        total_tokens: Optional[int] = None
        if usage is not None:
            total_attr = getattr(usage, "total_tokens", None)
            if total_attr is None and isinstance(usage, dict):
                total_attr = usage.get("total_tokens")
            if isinstance(total_attr, (int, float)):
                total_tokens = int(total_attr)
            else:
                input_tokens = getattr(usage, "input_tokens", None)
                output_tokens = getattr(usage, "output_tokens", None)
                if isinstance(usage, dict):
                    input_tokens = input_tokens if isinstance(input_tokens, (int, float)) else usage.get("input_tokens")
                    output_tokens = output_tokens if isinstance(output_tokens, (int, float)) else usage.get("output_tokens")
                total_tokens = int((input_tokens or 0) + (output_tokens or 0))
        if total_tokens is None:
            total_tokens = 0

        return {
            "success": True,
            "text": generated_text.strip(),
            "tokens_used": total_tokens
        }

    def generate_openai_text(self, prompt: str, model: str, temperature: float, max_tokens: int, system_prompt: str = None) -> Dict[str, Any]:
        """Generate text using OpenAI models."""
        if not self.openai_client:
            return {"success": False, "error": "OpenAI client not initialized"}

        try:
            return self._call_openai_responses(prompt, model, temperature, max_tokens, system_prompt)
            
        except Exception as e:
            error_str = str(e)
            return {"success": False, "error": f"Model {model}: {error_str}"}
    
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
                    'notes': row.get('notes', '')
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
                # Get universal system prompt with parameter-based instructions for human-like generation
                system_prompt = self.generator.get_universal_system_prompt(
                    temperature=item['temperature'],
                    max_tokens=item['max_tokens'],
                    style=item['style'],
                    domain=item['domain']
                )
                
                # Generate AI text with enhanced system prompt
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
