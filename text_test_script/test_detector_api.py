#!/usr/bin/env python3
"""
AI Detector API Testing Script

Simple script to test the AI detector API with a CSV file of labeled texts.
Calculates accuracy, precision, recall, and other performance metrics.

Usage:
    python test_detector_api.py --input test_data.csv --api-url http://localhost:8000
    python test_detector_api.py --input test_data.csv --api-url https://your-api.com --token your-token

CSV Format:
    text,label,model_name,word_count,sentence_count,domain,source,confidence,language
"""

import argparse
import csv
import json
import requests
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from tqdm import tqdm


class DetectorAPITester:
    """Simple API tester for AI detector endpoints."""
    
    def __init__(self, api_url: str, token: Optional[str] = None):
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
    
    def load_test_data(self, csv_file: str) -> List[Dict[str, Any]]:
        """Load test data from CSV file."""
        test_data = []
        
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Convert label to boolean for easier processing
                label = row['label'].lower().strip()
                is_ai_expected = label == 'ai' or label == 'true'
                
                test_data.append({
                    'text': row['text'],
                    'expected_label': label,
                    'is_ai_expected': is_ai_expected,
                    'metadata': {
                        'model_name': row.get('model_name', ''),
                        'word_count': int(row.get('word_count', 0)) if row.get('word_count') else 0,
                        'sentence_count': int(row.get('sentence_count', 0)) if row.get('sentence_count') else 0,
                        'domain': row.get('domain', ''),
                        'source': row.get('source', ''),
                        'confidence': float(row.get('confidence', 0)) if row.get('confidence') else 0,
                        'language': row.get('language', 'en')
                    }
                })
        
        print(f"Loaded {len(test_data)} test samples from {csv_file}")
        return test_data
    
    def call_api(self, text: str, detailed: bool = False) -> Dict[str, Any]:
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
    
    def run_tests(self, test_data: List[Dict[str, Any]], detailed: bool = False) -> None:
        """Run tests on all samples and collect results."""
        self.stats['total_samples'] = len(test_data)
        
        print(f"\nTesting {len(test_data)} samples...")
        print(f"API URL: {self.api_url}")
        print("-" * 50)
        
        for i, sample in enumerate(tqdm(test_data, desc="Testing samples")):
            result = self.call_api(sample['text'], detailed)
            
            # Track response time
            self.stats['total_response_time'] += result.get('response_time_ms', 0)
            
            if result.get('success'):
                self.stats['successful_predictions'] += 1
                
                # Calculate confusion matrix
                predicted_ai = result.get('is_ai', False)
                expected_ai = sample['is_ai_expected']
                
                if predicted_ai and expected_ai:
                    self.stats['true_positives'] += 1
                elif not predicted_ai and not expected_ai:
                    self.stats['true_negatives'] += 1
                elif predicted_ai and not expected_ai:
                    self.stats['false_positives'] += 1
                else:  # not predicted_ai and expected_ai
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
                    'metadata': sample['metadata']
                })
                
            else:
                self.stats['failed_requests'] += 1
                self.results.append({
                    'sample_id': i,
                    'text_preview': sample['text'][:100] + "...",
                    'expected_label': sample['expected_label'],
                    'error': result.get('error', 'Unknown error'),
                    'response_time_ms': result.get('response_time_ms', 0),
                    'metadata': sample['metadata']
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
            
            # Precision: TP / (TP + FP)
            metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
            
            # Recall: TP / (TP + FN)  
            metrics['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            # F1 Score
            p, r = metrics['precision'], metrics['recall']
            metrics['f1_score'] = 2 * (p * r) / (p + r) if (p + r) > 0 else 0
            
            # Average response time
            metrics['avg_response_time_ms'] = self.stats['total_response_time'] / total_predictions
            
        return metrics
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        metrics = self.calculate_metrics()
        
        report = {
            'test_summary': {
                'timestamp': datetime.now().isoformat(),
                'api_url': self.api_url,
                'total_samples': self.stats['total_samples'],
                'successful_predictions': self.stats['successful_predictions'],
                'failed_requests': self.stats['failed_requests'],
                **metrics
            },
            'confusion_matrix': {
                'true_positives': self.stats['true_positives'],
                'true_negatives': self.stats['true_negatives'],
                'false_positives': self.stats['false_positives'], 
                'false_negatives': self.stats['false_negatives']
            },
            'detailed_results': self.results
        }
        
        return report
    
    def print_summary(self) -> None:
        """Print test summary to console."""
        metrics = self.calculate_metrics()
        
        print("\n" + "="*60)
        print("TEST RESULTS SUMMARY")
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


def main():
    parser = argparse.ArgumentParser(description='Test AI Detector API with CSV data')
    parser.add_argument('--input', required=True, help='Input CSV file with test data')
    parser.add_argument('--api-url', default='http://localhost:8000', help='API base URL')
    parser.add_argument('--token', help='API authentication token')
    parser.add_argument('--output-dir', default='test_results', help='Output directory for results')
    parser.add_argument('--detailed', action='store_true', help='Request detailed sentence-level analysis')
    parser.add_argument('--quiet', action='store_true', help='Minimal console output')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Initialize tester
    tester = DetectorAPITester(args.api_url, args.token)
    
    try:
        # Load test data
        test_data = tester.load_test_data(args.input)
        
        # Run tests
        tester.run_tests(test_data, args.detailed)
        
        # Generate report
        report = tester.generate_report()
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = output_dir / f'test_results_{timestamp}.json'
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # Print summary
        if not args.quiet:
            tester.print_summary()
        
        print(f"\nDetailed results saved to: {output_file}")
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())