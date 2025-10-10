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
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from io import StringIO, BytesIO
from tqdm import tqdm
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv(".env")

# Try to import boto3 for S3 support
try:
    import boto3
    from botocore.exceptions import ClientError
    S3_AVAILABLE = True
except ImportError:
    S3_AVAILABLE = False
    print("⚠️ boto3 not installed. S3 support will be disabled.")

# Try to import OpenTelemetry for metrics export
try:
    from opentelemetry import metrics
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.exporter.prometheus_remote_write import (
        PrometheusRemoteWriteMetricsExporter
    )
    from opentelemetry.sdk.resources import Resource, SERVICE_NAME
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    print("⚠️ OpenTelemetry not installed. Metrics export will be disabled.")


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


class S3Helper:
    """Helper class for S3 operations."""
    
    def __init__(self, bucket_name: str = None, prefix: str = ""):
        self.bucket_name = bucket_name or os.getenv('S3_BUCKET_NAME')
        self.prefix = prefix.strip('/') + '/' if prefix else ''
        self.s3_client = None
        
        if S3_AVAILABLE and self.bucket_name:
            try:
                self.s3_client = boto3.client('s3')
                print(f"✅ S3 client initialized (bucket: {self.bucket_name})")
            except Exception as e:
                print(f"⚠️ S3 client initialization failed: {e}")
    
    def is_available(self) -> bool:
        """Check if S3 is available and configured."""
        return self.s3_client is not None and self.bucket_name is not None
    
    def upload_file(self, local_path: str, s3_key: str = None) -> bool:
        """Upload a local file to S3."""
        if not self.is_available():
            return False
        
        try:
            if s3_key is None:
                s3_key = self.prefix + Path(local_path).name
            else:
                s3_key = self.prefix + s3_key
            
            self.s3_client.upload_file(local_path, self.bucket_name, s3_key)
            print(f"✅ Uploaded to S3: s3://{self.bucket_name}/{s3_key}")
            return True
        except ClientError as e:
            print(f"❌ Failed to upload {local_path} to S3: {e}")
            return False
    
    def upload_string(self, content: str, s3_key: str, content_type: str = 'text/plain') -> bool:
        """Upload string content directly to S3."""
        if not self.is_available():
            return False
        
        try:
            s3_key = self.prefix + s3_key
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=content.encode('utf-8'),
                ContentType=content_type
            )
            print(f"✅ Uploaded to S3: s3://{self.bucket_name}/{s3_key}")
            return True
        except ClientError as e:
            print(f"❌ Failed to upload content to S3: {e}")
            return False
    
    def upload_json(self, data: dict, s3_key: str) -> bool:
        """Upload JSON data to S3."""
        content = json.dumps(data, indent=2, ensure_ascii=False)
        return self.upload_string(content, s3_key, content_type='application/json')
    
    def download_to_string(self, s3_key: str) -> Optional[str]:
        """Download S3 object to string."""
        if not self.is_available():
            return None
        
        try:
            s3_key = self.prefix + s3_key
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=s3_key)
            return response['Body'].read().decode('utf-8')
        except ClientError as e:
            print(f"❌ Failed to download from S3: {e}")
            return None


class PromptBasedTester:
    """Main class that orchestrates generation and testing."""
    
    def __init__(self, api_url: str = None, token: str = None, s3_bucket: str = None, s3_prefix: str = ""):
        self.generator = AITextGenerator()
        self.tester = DetectorAPITester(
            api_url or os.getenv('DETECTOR_API_URL', 'http://localhost:8000'),
            token or os.getenv('DETECTOR_API_TOKEN')
        )
        self.s3_helper = S3Helper(s3_bucket, s3_prefix)
        self.generated_data = []
        
        # Initialize OpenTelemetry metrics if available
        self.meter = None
        self.metrics_exporter = None
        if OTEL_AVAILABLE:
            self._init_otel_metrics()
    
    def _init_otel_metrics(self):
        """Initialize OpenTelemetry metrics provider and exporter."""
        try:
            remote_write_url = os.getenv('PROMETHEUS_REMOTE_WRITE_URL')
            if not remote_write_url:
                print("⚠️ PROMETHEUS_REMOTE_WRITE_URL not set, metrics export disabled")
                return
            
            # Create resource with service information
            resource = Resource.create({
                SERVICE_NAME: "ai-detector-eval",
                "environment": os.getenv("ENVIRONMENT", "local"),
                "cluster": os.getenv("CLUSTER", "local-test"),
            })
            
            # Configure Prometheus Remote Write exporter
            self.metrics_exporter = PrometheusRemoteWriteMetricsExporter(
                endpoint=remote_write_url,
                headers={
                    "X-Scope-OrgID": os.getenv("PROMETHEUS_TENANT_ID", "anonymous")
                },
                timeout=30
            )
            
            # Create metric reader with periodic export
            reader = PeriodicExportingMetricReader(
                exporter=self.metrics_exporter,
                export_interval_millis=5000  # Export every 5 seconds
            )
            
            # Create and set meter provider
            provider = MeterProvider(
                resource=resource,
                metric_readers=[reader]
            )
            
            metrics.set_meter_provider(provider)
            self.meter = metrics.get_meter("ai_detector_eval", "1.0.0")
            
            print("✅ OpenTelemetry metrics initialized")
        except Exception as e:
            print(f"⚠️ Failed to initialize OpenTelemetry metrics: {e}")
            self.meter = None
    
    def load_prompts(self, csv_file: str) -> List[Dict[str, Any]]:
        """Load prompts and human texts from local CSV file."""
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
    
    def load_prompts_from_s3(self, s3_path: str) -> List[Dict[str, Any]]:
        """Load prompts and human texts from S3 CSV file.
        
        Args:
            s3_path: S3 path in format 's3://bucket/key' or 'bucket/key'
        
        Returns:
            List of prompt dictionaries
        """
        if not self.s3_helper.is_available():
            raise RuntimeError("S3 not available. Install boto3 and configure S3_BUCKET_NAME")
        
        # Parse S3 path
        if s3_path.startswith('s3://'):
            s3_path = s3_path[5:]  # Remove 's3://' prefix
        
        parts = s3_path.split('/', 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid S3 path format: {s3_path}. Expected 's3://bucket/key' or 'bucket/key'")
        
        bucket, key = parts
        
        # Download from S3
        print(f"Downloading prompts from s3://{bucket}/{key}...")
        content = self.s3_helper.s3_client.get_object(Bucket=bucket, Key=key)['Body'].read().decode('utf-8')
        
        if not content:
            raise ValueError(f"Empty file downloaded from s3://{bucket}/{key}")
        
        # Parse CSV from string
        prompts_data = []
        reader = csv.DictReader(StringIO(content))
        
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
        
        print(f"✅ Loaded {len(prompts_data)} items from s3://{bucket}/{key}")
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
        
        # Also upload to S3 if available
        if self.s3_helper.is_available():
            self.s3_helper.upload_file(output_file)
    
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
        
        # Also upload to S3 if available
        if self.s3_helper.is_available():
            self.s3_helper.upload_file(output_file)
    
    def save_results_jsonl(self, output_file: str) -> None:
        """Save results in JSONL format for streaming/S3 storage."""
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in self.tester.results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        print(f"JSONL results saved to: {output_file}")
        
        # Also upload to S3 if available
        if self.s3_helper.is_available():
            self.s3_helper.upload_file(output_file)
    
    def generate_manifest(self, output_file: str, run_id: str = None, 
                         dataset_version: str = None, commit_sha: str = None) -> None:
        """Generate manifest file with run metadata."""
        metrics = self.tester.calculate_metrics()
        
        manifest = {
            'run_id': run_id or os.getenv('RUN_ID', str(int(time.time()))),
            'timestamp': datetime.now().isoformat(),
            'dataset_version': dataset_version or os.getenv('DATASET_VERSION', 'unknown'),
            'commit_sha': commit_sha or os.getenv('GIT_COMMIT_SHA', 'unknown'),
            'detector_api_url': self.tester.api_url,
            'total_samples': self.tester.stats['total_samples'],
            'sample_breakdown': {
                'ai_generated': len([r for r in self.tester.results 
                                    if r.get('expected_label') == 'ai']),
                'human_written': len([r for r in self.tester.results 
                                     if r.get('expected_label') == 'human'])
            },
            'metrics': metrics,
            'confusion_matrix': {
                'true_positives': self.tester.stats['true_positives'],
                'true_negatives': self.tester.stats['true_negatives'],
                'false_positives': self.tester.stats['false_positives'],
                'false_negatives': self.tester.stats['false_negatives']
            },
            'models_tested': list(set([
                r.get('metadata', {}).get('model', 'unknown') 
                for r in self.tester.results 
                if r.get('metadata', {}).get('type') == 'generated'
            ])),
            'failed_requests': self.tester.stats['failed_requests']
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        
        print(f"Manifest saved to: {output_file}")
        
        # Also upload to S3 if available
        if self.s3_helper.is_available():
            self.s3_helper.upload_file(output_file)
    
    def export_metrics_to_mimir(self, run_duration: float = 0) -> None:
        """Export metrics to Mimir via OpenTelemetry Remote Write."""
        if not OTEL_AVAILABLE:
            print("⚠️ OpenTelemetry not available, skipping metrics export")
            return
        
        if not self.meter:
            print("⚠️ Metrics meter not initialized, skipping export")
            return
        
        try:
            metrics_data = self.tester.calculate_metrics()
            
            # Create observable gauges for current state metrics
            # Note: In OTEL, we use callbacks for gauges to get current values
            
            def success_callback(options):
                """Callback for success status gauge."""
                yield metrics.Observation(
                    1 if self.tester.stats['failed_requests'] == 0 else 0
                )
            
            def timestamp_callback(options):
                """Callback for timestamp gauge."""
                yield metrics.Observation(time.time())
            
            def duration_callback(options):
                """Callback for duration gauge."""
                yield metrics.Observation(run_duration)
            
            def accuracy_callback(options):
                """Callback for accuracy gauge."""
                yield metrics.Observation(metrics_data.get('accuracy', 0))
            
            def precision_callback(options):
                """Callback for precision gauge."""
                yield metrics.Observation(metrics_data.get('precision', 0))
            
            def recall_callback(options):
                """Callback for recall gauge."""
                yield metrics.Observation(metrics_data.get('recall', 0))
            
            def f1_callback(options):
                """Callback for F1 score gauge."""
                yield metrics.Observation(metrics_data.get('f1_score', 0))
            
            # Create observable gauges (these are exported automatically)
            success_gauge = self.meter.create_observable_gauge(
                name="ai_detector_eval_success",
                callbacks=[success_callback],
                description="Evaluation run success status (1=success, 0=failure)",
                unit="1"
            )
            
            timestamp_gauge = self.meter.create_observable_gauge(
                name="ai_detector_eval_run_timestamp_seconds",
                callbacks=[timestamp_callback],
                description="Timestamp of evaluation run",
                unit="s"
            )
            
            duration_gauge = self.meter.create_observable_gauge(
                name="ai_detector_eval_duration_seconds",
                callbacks=[duration_callback],
                description="Duration of evaluation run",
                unit="s"
            )
            
            accuracy_gauge = self.meter.create_observable_gauge(
                name="ai_detector_eval_accuracy",
                callbacks=[accuracy_callback],
                description="Overall accuracy",
                unit="1"
            )
            
            precision_gauge = self.meter.create_observable_gauge(
                name="ai_detector_eval_precision",
                callbacks=[precision_callback],
                description="Precision metric",
                unit="1"
            )
            
            recall_gauge = self.meter.create_observable_gauge(
                name="ai_detector_eval_recall",
                callbacks=[recall_callback],
                description="Recall metric",
                unit="1"
            )
            
            f1_gauge = self.meter.create_observable_gauge(
                name="ai_detector_eval_f1",
                callbacks=[f1_callback],
                description="F1 score",
                unit="1"
            )
            
            # For metrics with labels (attributes in OTEL), we need counters
            # Sample counts by label
            ai_count = self.tester.stats['true_positives'] + self.tester.stats['false_negatives']
            human_count = self.tester.stats['true_negatives'] + self.tester.stats['false_positives']
            
            samples_counter = self.meter.create_counter(
                name="ai_detector_eval_samples_total",
                description="Total samples evaluated",
                unit="1"
            )
            samples_counter.add(ai_count, {"label": "ai"})
            samples_counter.add(human_count, {"label": "human"})
            
            # Confusion matrix metrics
            confusion_counter = self.meter.create_counter(
                name="ai_detector_eval_confusion_matrix",
                description="Confusion matrix counts",
                unit="1"
            )
            confusion_counter.add(self.tester.stats['true_positives'], {"kind": "tp"})
            confusion_counter.add(self.tester.stats['true_negatives'], {"kind": "tn"})
            confusion_counter.add(self.tester.stats['false_positives'], {"kind": "fp"})
            confusion_counter.add(self.tester.stats['false_negatives'], {"kind": "fn"})
            
            # Force a flush of metrics to ensure they're exported
            if self.metrics_exporter:
                print("✅ Metrics exported to Mimir via OpenTelemetry")
            
        except Exception as e:
            print(f"⚠️ Failed to export metrics: {e}")
            import traceback
            traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description='AI Detector Prompt-Based Testing')
    parser.add_argument('--input', help='Input CSV file with prompts (local path or s3://bucket/key)')
    parser.add_argument('--s3-input', action='store_true', help='Treat --input as S3 path (s3://bucket/key)')
    parser.add_argument('--api-url', help='Detector API base URL (or use DETECTOR_API_URL env var)')
    parser.add_argument('--token', help='Detector API authentication token (or use DETECTOR_API_TOKEN env var)')
    parser.add_argument('--output-dir', default='test_results', help='Output directory for results')
    parser.add_argument('--detailed', action='store_true', help='Request detailed sentence-level analysis')
    parser.add_argument('--generate-only', action='store_true', help='Only generate texts, don\'t test')
    parser.add_argument('--test-only', action='store_true', help='Only test (input should be generated test data CSV)')
    parser.add_argument('--generate-and-test', action='store_true', help='Generate texts and test them')
    parser.add_argument('--quiet', action='store_true', help='Minimal console output')
    
    # Production/automation arguments
    parser.add_argument('--run-id', help='Unique run identifier (for tracking)')
    parser.add_argument('--dataset-version', help='Dataset version label')
    parser.add_argument('--commit-sha', help='Git commit SHA for tracking')
    parser.add_argument('--save-jsonl', action='store_true', help='Save results in JSONL format')
    parser.add_argument('--save-manifest', action='store_true', help='Generate manifest.json with run metadata')
    parser.add_argument('--exit-on-error', action='store_true', help='Exit with error code if any tests fail')
    
    # S3 and serverless arguments
    parser.add_argument('--s3-bucket', help='S3 bucket name for output files (or use S3_BUCKET_NAME env var)')
    parser.add_argument('--s3-prefix', default='', help='S3 key prefix for uploaded files')
    
    args = parser.parse_args()
    
    # Default to generate-and-test if no mode specified
    if not any([args.generate_only, args.test_only, args.generate_and_test]):
        args.generate_and_test = True
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Initialize tester with S3 support
    tester = PromptBasedTester(args.api_url, args.token, args.s3_bucket, args.s3_prefix)
    
    # Track run duration for metrics
    start_time = time.time()
    exit_code = 0
    
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if args.generate_only or args.generate_and_test:
            # Load prompts from S3 or local file
            if args.s3_input:
                if not args.input:
                    print("❌ Error: --input required when using --s3-input")
                    return 1
                prompts_data = tester.load_prompts_from_s3(args.input)
            else:
                if not args.input:
                    print("❌ Error: --input required")
                    return 1
                prompts_data = tester.load_prompts(args.input)
            
            # Generate texts
            test_data = tester.generate_texts(prompts_data)
            
            # Save generated data
            generated_file = output_dir / f'generated_test_data_{timestamp}.csv'
            tester.save_generated_data(str(generated_file))
            
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
            
            # Calculate run duration
            run_duration = time.time() - start_time
            
            # Generate report
            report_file = output_dir / f'prompt_test_report_{timestamp}.json'
            tester.generate_report(str(report_file))
            
            # NEW: Save JSONL format if requested
            if args.save_jsonl:
                jsonl_file = output_dir / f'results_{timestamp}.jsonl'
                tester.save_results_jsonl(str(jsonl_file))
            
            # NEW: Generate manifest if requested
            if args.save_manifest:
                manifest_file = output_dir / f'manifest_{timestamp}.json'
                tester.generate_manifest(
                    str(manifest_file),
                    run_id=args.run_id,
                    dataset_version=args.dataset_version,
                    commit_sha=args.commit_sha
                )
            
            # Export metrics to Mimir via OpenTelemetry
            tester.export_metrics_to_mimir(run_duration=run_duration)
            
            # Check for failures and set exit code
            if args.exit_on_error and tester.tester.stats['failed_requests'] > 0:
                print(f"\n❌ {tester.tester.stats['failed_requests']} requests failed")
                exit_code = 1
        
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
        exit_code = 130  # Standard exit code for SIGINT
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        exit_code = 1
    
    return exit_code


if __name__ == '__main__':
    sys.exit(main())
