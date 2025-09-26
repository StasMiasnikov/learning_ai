# Performance Measurement: Evaluation Metrics and Monitoring

Measuring the performance of LLM agents and multi-agent systems requires comprehensive metrics that capture accuracy, efficiency, reliability, and user experience. This section covers essential evaluation frameworks and monitoring strategies.

## 📊 Core Performance Metrics

### LLM Quality Metrics

**Accuracy and Correctness**:
```python
import numpy as np
from typing import List, Dict, Any, Tuple
import re
from collections import Counter

class LLMQualityMetrics:
    """Comprehensive quality metrics for LLM outputs"""
    
    def __init__(self):
        self.fact_checker = FactualAccuracyChecker()
        self.coherence_scorer = CoherenceScorer()
        self.bias_detector = BiasDetector()
    
    def evaluate_response(self, 
                         response: str, 
                         reference: str = None, 
                         context: Dict[str, Any] = None) -> Dict[str, float]:
        """Comprehensive response evaluation"""
        
        metrics = {}
        
        # Basic text quality metrics
        metrics.update(self._basic_text_metrics(response))
        
        # Content quality metrics
        if reference:
            metrics.update(self._content_similarity_metrics(response, reference))
        
        # Coherence and fluency
        metrics['coherence_score'] = self.coherence_scorer.score(response)
        metrics['fluency_score'] = self._calculate_fluency(response)
        
        # Factual accuracy (if context provided)
        if context and 'domain' in context:
            metrics['factual_accuracy'] = self.fact_checker.check_accuracy(
                response, context['domain']
            )
        
        # Bias detection
        metrics['bias_score'] = self.bias_detector.detect_bias(response)
        
        # Safety and appropriateness
        metrics['safety_score'] = self._calculate_safety_score(response)
        
        return metrics
    
    def _basic_text_metrics(self, text: str) -> Dict[str, float]:
        """Calculate basic text quality metrics"""
        words = text.split()
        sentences = text.split('.')
        
        return {
            'word_count': len(words),
            'sentence_count': len([s for s in sentences if s.strip()]),
            'avg_word_length': np.mean([len(word) for word in words]) if words else 0,
            'avg_sentence_length': len(words) / max(len(sentences) - 1, 1),
            'lexical_diversity': len(set(words)) / max(len(words), 1)
        }
    
    def _content_similarity_metrics(self, response: str, reference: str) -> Dict[str, float]:
        """Calculate content similarity metrics"""
        # BLEU score approximation
        bleu = self._calculate_bleu(response, reference)
        
        # ROUGE scores
        rouge_scores = self._calculate_rouge(response, reference)
        
        # Semantic similarity (simplified)
        semantic_sim = self._semantic_similarity(response, reference)
        
        return {
            'bleu_score': bleu,
            'rouge_1': rouge_scores['rouge_1'],
            'rouge_2': rouge_scores['rouge_2'],
            'rouge_l': rouge_scores['rouge_l'],
            'semantic_similarity': semantic_sim
        }
    
    def _calculate_bleu(self, candidate: str, reference: str, max_n: int = 4) -> float:
        """Calculate BLEU score"""
        def get_ngrams(text: str, n: int) -> List[str]:
            words = text.lower().split()
            return [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]
        
        candidate_words = candidate.lower().split()
        reference_words = reference.lower().split()
        
        if not candidate_words:
            return 0.0
        
        precisions = []
        for n in range(1, max_n + 1):
            cand_ngrams = get_ngrams(candidate, n)
            ref_ngrams = get_ngrams(reference, n)
            
            if not cand_ngrams:
                precisions.append(0.0)
                continue
            
            cand_counter = Counter(cand_ngrams)
            ref_counter = Counter(ref_ngrams)
            
            overlap = sum((cand_counter & ref_counter).values())
            precision = overlap / len(cand_ngrams)
            precisions.append(precision)
        
        if not all(p > 0 for p in precisions):
            return 0.0
        
        # Geometric mean
        bleu = np.exp(np.mean(np.log(precisions)))
        
        # Brevity penalty
        bp = min(1.0, np.exp(1 - len(reference_words) / len(candidate_words)))
        
        return bleu * bp
    
    def _calculate_rouge(self, candidate: str, reference: str) -> Dict[str, float]:
        """Calculate ROUGE scores"""
        def rouge_n(cand: str, ref: str, n: int) -> float:
            cand_ngrams = [' '.join(cand.lower().split()[i:i+n]) 
                          for i in range(len(cand.split())-n+1)]
            ref_ngrams = [' '.join(ref.lower().split()[i:i+n]) 
                         for i in range(len(ref.split())-n+1)]
            
            if not ref_ngrams:
                return 0.0
            
            overlap = len(set(cand_ngrams) & set(ref_ngrams))
            return overlap / len(ref_ngrams)
        
        def rouge_l(cand: str, ref: str) -> float:
            """ROUGE-L (Longest Common Subsequence)"""
            cand_words = cand.lower().split()
            ref_words = ref.lower().split()
            
            # Simple LCS implementation
            m, n = len(cand_words), len(ref_words)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if cand_words[i-1] == ref_words[j-1]:
                        dp[i][j] = dp[i-1][j-1] + 1
                    else:
                        dp[i][j] = max(dp[i-1][j], dp[i][j-1])
            
            lcs_length = dp[m][n]
            return lcs_length / max(n, 1)
        
        return {
            'rouge_1': rouge_n(candidate, reference, 1),
            'rouge_2': rouge_n(candidate, reference, 2),
            'rouge_l': rouge_l(candidate, reference)
        }
    
    def _semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity (simplified word overlap)"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / max(union, 1)
    
    def _calculate_fluency(self, text: str) -> float:
        """Calculate fluency score based on linguistic patterns"""
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        if not sentences:
            return 0.0
        
        fluency_indicators = []
        
        for sentence in sentences:
            words = sentence.split()
            
            # Sentence length (optimal around 15-20 words)
            length_score = 1.0 - abs(len(words) - 17.5) / 17.5
            length_score = max(0, min(1, length_score))
            
            # Grammar indicators (simplified)
            has_subject_verb = any(word.lower() in ['is', 'are', 'was', 'were', 'have', 'has'] for word in words)
            grammar_score = 0.8 if has_subject_verb else 0.3
            
            # Punctuation appropriateness
            punct_score = 0.9 if sentence.endswith(('.', '!', '?')) else 0.5
            
            sentence_fluency = np.mean([length_score, grammar_score, punct_score])
            fluency_indicators.append(sentence_fluency)
        
        return np.mean(fluency_indicators)
    
    def _calculate_safety_score(self, text: str) -> float:
        """Calculate safety score (absence of harmful content)"""
        harmful_patterns = [
            r'\bhate\b', r'\bkill\b', r'\bhurt\b', r'\battack\b',
            r'\bviolence\b', r'\bharm\b', r'\babuse\b'
        ]
        
        safety_violations = 0
        for pattern in harmful_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                safety_violations += 1
        
        # Safety score decreases with violations
        return max(0.0, 1.0 - (safety_violations * 0.2))

class FactualAccuracyChecker:
    """Check factual accuracy of LLM responses"""
    
    def __init__(self):
        # Domain-specific fact databases (simplified)
        self.known_facts = {
            'science': {
                'water boils at 100 celsius': True,
                'earth is flat': False,
                'gravity exists': True,
            },
            'history': {
                'world war 2 ended in 1945': True,
                'napoleon was born in 1769': True,
                'columbus discovered america in 1492': True,
            }
        }
    
    def check_accuracy(self, text: str, domain: str) -> float:
        """Check factual accuracy against known facts"""
        if domain not in self.known_facts:
            return 0.5  # Unknown domain
        
        domain_facts = self.known_facts[domain]
        text_lower = text.lower()
        
        fact_checks = []
        for fact, is_true in domain_facts.items():
            if fact in text_lower:
                fact_checks.append(1.0 if is_true else 0.0)
        
        return np.mean(fact_checks) if fact_checks else 0.5

class CoherenceScorer:
    """Score text coherence and logical flow"""
    
    def score(self, text: str) -> float:
        """Score overall coherence"""
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        if len(sentences) < 2:
            return 0.5
        
        coherence_scores = []
        
        # Sentence transitions
        transition_score = self._score_transitions(sentences)
        coherence_scores.append(transition_score)
        
        # Topic consistency
        topic_score = self._score_topic_consistency(sentences)
        coherence_scores.append(topic_score)
        
        # Logical flow
        logic_score = self._score_logical_flow(sentences)
        coherence_scores.append(logic_score)
        
        return np.mean(coherence_scores)
    
    def _score_transitions(self, sentences: List[str]) -> float:
        """Score sentence transitions"""
        transition_words = [
            'however', 'therefore', 'moreover', 'furthermore', 'additionally',
            'consequently', 'meanwhile', 'similarly', 'in contrast', 'for example'
        ]
        
        transition_count = 0
        for sentence in sentences:
            if any(word in sentence.lower() for word in transition_words):
                transition_count += 1
        
        # Optimal: 20-40% sentences with transitions
        ratio = transition_count / len(sentences)
        optimal_ratio = 0.3
        
        return 1.0 - abs(ratio - optimal_ratio) / optimal_ratio
    
    def _score_topic_consistency(self, sentences: List[str]) -> float:
        """Score topic consistency across sentences"""
        all_words = []
        sentence_words = []
        
        for sentence in sentences:
            words = [w.lower() for w in sentence.split() if len(w) > 3]
            sentence_words.append(set(words))
            all_words.extend(words)
        
        if not all_words:
            return 0.5
        
        # Calculate word overlap between consecutive sentences
        overlaps = []
        for i in range(len(sentence_words) - 1):
            overlap = len(sentence_words[i] & sentence_words[i + 1])
            overlaps.append(overlap / max(len(sentence_words[i]), 1))
        
        return np.mean(overlaps) if overlaps else 0.5
    
    def _score_logical_flow(self, sentences: List[str]) -> float:
        """Score logical flow (simplified)"""
        # Check for logical connectors and argument structure
        logical_indicators = [
            'because', 'since', 'therefore', 'thus', 'consequently',
            'if', 'then', 'although', 'despite', 'while'
        ]
        
        logic_count = 0
        for sentence in sentences:
            if any(indicator in sentence.lower() for indicator in logical_indicators):
                logic_count += 1
        
        return min(1.0, logic_count / max(len(sentences), 1))

class BiasDetector:
    """Detect potential bias in text"""
    
    def __init__(self):
        self.bias_terms = {
            'gender': ['he said', 'she said', 'guys', 'ladies'],
            'racial': ['black', 'white', 'asian', 'hispanic'],
            'age': ['young', 'old', 'elderly', 'teenager'],
            'religious': ['christian', 'muslim', 'jewish', 'atheist']
        }
    
    def detect_bias(self, text: str) -> float:
        """Detect bias indicators (simplified approach)"""
        text_lower = text.lower()
        bias_indicators = 0
        
        for category, terms in self.bias_terms.items():
            for term in terms:
                if term in text_lower:
                    # Check for stereotypical associations
                    bias_indicators += self._check_stereotypical_context(text_lower, term)
        
        # Return bias score (lower is better)
        return min(1.0, bias_indicators * 0.1)
    
    def _check_stereotypical_context(self, text: str, term: str) -> int:
        """Check if term appears in stereotypical context"""
        stereotypical_words = [
            'always', 'never', 'all', 'typical', 'usually', 'naturally'
        ]
        
        # Simple context window check
        term_pos = text.find(term)
        if term_pos == -1:
            return 0
        
        context_start = max(0, term_pos - 50)
        context_end = min(len(text), term_pos + len(term) + 50)
        context = text[context_start:context_end]
        
        return sum(1 for word in stereotypical_words if word in context)

# Example usage
evaluator = LLMQualityMetrics()

test_responses = [
    {
        'response': "Artificial intelligence is a rapidly growing field that involves creating systems capable of performing tasks that typically require human intelligence. These systems use machine learning algorithms to process data and make decisions.",
        'reference': "AI is a field focused on creating intelligent systems using machine learning to process data and make decisions.",
        'context': {'domain': 'science'}
    },
    {
        'response': "The earth is flat and gravity doesn't exist. All scientists are lying to us about basic physics.",
        'reference': "The earth is spherical and gravity is a fundamental force in physics.",
        'context': {'domain': 'science'}
    }
]

for i, test in enumerate(test_responses):
    print(f"\nEvaluating Response {i + 1}:")
    print(f"Response: {test['response'][:100]}...")
    
    metrics = evaluator.evaluate_response(
        test['response'], 
        test.get('reference'), 
        test.get('context')
    )
    
    print(f"Quality Metrics:")
    for metric, value in metrics.items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.3f}")
        else:
            print(f"  {metric}: {value}")
```

### Performance and Efficiency Metrics

**Response Time and Throughput**:
```python
import time
import asyncio
from typing import List, Dict, Any
import statistics
from collections import deque
import threading

class PerformanceMonitor:
    """Monitor LLM system performance and efficiency"""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.response_times = deque(maxlen=window_size)
        self.token_counts = deque(maxlen=window_size)
        self.request_timestamps = deque(maxlen=window_size)
        self.error_counts = {'total': 0, 'timeout': 0, 'server_error': 0, 'rate_limit': 0}
        self.lock = threading.Lock()
    
    def record_request(self, response_time: float, token_count: int, error_type: str = None):
        """Record a request's performance metrics"""
        with self.lock:
            current_time = time.time()
            
            if error_type:
                self.error_counts['total'] += 1
                self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
            else:
                self.response_times.append(response_time)
                self.token_counts.append(token_count)
            
            self.request_timestamps.append(current_time)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        with self.lock:
            if not self.response_times:
                return {'status': 'no_data'}
            
            current_time = time.time()
            
            # Response time statistics
            response_times_list = list(self.response_times)
            response_stats = {
                'mean_response_time': statistics.mean(response_times_list),
                'median_response_time': statistics.median(response_times_list),
                'p95_response_time': self._percentile(response_times_list, 95),
                'p99_response_time': self._percentile(response_times_list, 99),
                'min_response_time': min(response_times_list),
                'max_response_time': max(response_times_list)
            }
            
            # Throughput calculations
            recent_requests = [ts for ts in self.request_timestamps 
                             if current_time - ts <= 3600]  # Last hour
            
            throughput_stats = {
                'requests_per_hour': len(recent_requests),
                'requests_per_minute': len([ts for ts in recent_requests 
                                          if current_time - ts <= 60]),
                'requests_per_second': len([ts for ts in recent_requests 
                                          if current_time - ts <= 1])
            }
            
            # Token efficiency
            token_list = list(self.token_counts)
            token_stats = {
                'mean_tokens_per_request': statistics.mean(token_list),
                'tokens_per_second': sum(token_list) / sum(response_times_list) if response_times_list else 0,
                'total_tokens_processed': sum(token_list)
            }
            
            # Error rates
            total_requests = len(self.request_timestamps)
            error_stats = {
                'error_rate': self.error_counts['total'] / max(total_requests, 1),
                'timeout_rate': self.error_counts.get('timeout', 0) / max(total_requests, 1),
                'server_error_rate': self.error_counts.get('server_error', 0) / max(total_requests, 1)
            }
            
            return {
                'response_time': response_stats,
                'throughput': throughput_stats,
                'token_efficiency': token_stats,
                'error_rates': error_stats,
                'total_requests': total_requests,
                'timestamp': current_time
            }
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile of data"""
        sorted_data = sorted(data)
        index = int((percentile / 100) * len(sorted_data))
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    def get_system_health(self) -> Dict[str, str]:
        """Get overall system health status"""
        metrics = self.get_performance_metrics()
        
        if metrics.get('status') == 'no_data':
            return {'health': 'unknown', 'message': 'Insufficient data'}
        
        health_indicators = []
        
        # Response time health
        if metrics['response_time']['p95_response_time'] > 5.0:
            health_indicators.append('slow_response')
        
        # Error rate health
        if metrics['error_rates']['error_rate'] > 0.05:  # 5% error rate
            health_indicators.append('high_error_rate')
        
        # Throughput health
        if metrics['throughput']['requests_per_minute'] < 1:
            health_indicators.append('low_throughput')
        
        if not health_indicators:
            return {'health': 'healthy', 'message': 'All systems operational'}
        elif len(health_indicators) == 1:
            return {'health': 'warning', 'message': f'Issue detected: {health_indicators[0]}'}
        else:
            return {'health': 'critical', 'message': f'Multiple issues: {", ".join(health_indicators)}'}

class LoadTester:
    """Load testing utility for LLM systems"""
    
    def __init__(self, target_function, monitor: PerformanceMonitor):
        self.target_function = target_function
        self.monitor = monitor
    
    async def run_load_test(self, 
                           concurrent_users: int = 10,
                           requests_per_user: int = 10,
                           test_duration: int = 60) -> Dict[str, Any]:
        """Run comprehensive load test"""
        
        print(f"Starting load test: {concurrent_users} users, {requests_per_user} requests each")
        
        # Test scenarios
        test_queries = [
            "What is artificial intelligence?",
            "Explain machine learning in simple terms.",
            "How do neural networks work?",
            "What are the benefits of AI?",
            "Describe the history of computers."
        ]
        
        async def user_session(user_id: int):
            """Simulate user session"""
            user_results = []
            
            for request_num in range(requests_per_user):
                query = test_queries[request_num % len(test_queries)]
                
                start_time = time.time()
                try:
                    # Simulate LLM call
                    response = await self.simulate_llm_call(query)
                    end_time = time.time()
                    
                    response_time = end_time - start_time
                    token_count = len(response.split()) * 1.3  # Approximate token count
                    
                    self.monitor.record_request(response_time, int(token_count))
                    user_results.append({
                        'success': True,
                        'response_time': response_time,
                        'tokens': int(token_count)
                    })
                    
                except Exception as e:
                    end_time = time.time()
                    self.monitor.record_request(0, 0, 'server_error')
                    user_results.append({
                        'success': False,
                        'error': str(e),
                        'response_time': end_time - start_time
                    })
                
                # Random delay between requests
                await asyncio.sleep(np.random.uniform(0.1, 2.0))
            
            return user_results
        
        # Run concurrent user sessions
        start_time = time.time()
        tasks = [user_session(i) for i in range(concurrent_users)]
        all_results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = time.time()
        
        # Aggregate results
        successful_requests = 0
        failed_requests = 0
        total_response_time = 0
        
        for user_results in all_results:
            if isinstance(user_results, Exception):
                failed_requests += requests_per_user
                continue
                
            for result in user_results:
                if result['success']:
                    successful_requests += 1
                    total_response_time += result['response_time']
                else:
                    failed_requests += 1
        
        total_requests = successful_requests + failed_requests
        
        return {
            'test_duration': end_time - start_time,
            'total_requests': total_requests,
            'successful_requests': successful_requests,
            'failed_requests': failed_requests,
            'success_rate': successful_requests / max(total_requests, 1),
            'average_response_time': total_response_time / max(successful_requests, 1),
            'requests_per_second': total_requests / (end_time - start_time),
            'concurrent_users': concurrent_users,
            'performance_metrics': self.monitor.get_performance_metrics()
        }
    
    async def simulate_llm_call(self, query: str) -> str:
        """Simulate LLM API call with realistic timing"""
        # Simulate processing time based on query length
        base_time = 0.5  # Base response time
        query_factor = len(query.split()) * 0.1  # Additional time per word
        
        processing_time = base_time + query_factor + np.random.uniform(0, 1)
        await asyncio.sleep(processing_time)
        
        # Simulate occasional failures
        if np.random.random() < 0.02:  # 2% failure rate
            raise Exception("Simulated API error")
        
        # Generate mock response
        response_length = np.random.randint(50, 200)  # Random response length
        return " ".join(["word"] * response_length)

# Example usage
import numpy as np

async def demo_performance_monitoring():
    """Demonstrate performance monitoring"""
    
    monitor = PerformanceMonitor()
    
    # Simulate some requests
    for i in range(50):
        response_time = np.random.uniform(0.5, 3.0)
        token_count = np.random.randint(50, 500)
        
        # Occasional errors
        if np.random.random() < 0.05:
            monitor.record_request(0, 0, 'server_error')
        else:
            monitor.record_request(response_time, token_count)
        
        await asyncio.sleep(0.1)  # Small delay
    
    # Get performance metrics
    metrics = monitor.get_performance_metrics()
    health = monitor.get_system_health()
    
    print("Performance Metrics:")
    print(f"Mean Response Time: {metrics['response_time']['mean_response_time']:.3f}s")
    print(f"P95 Response Time: {metrics['response_time']['p95_response_time']:.3f}s")
    print(f"Requests/Minute: {metrics['throughput']['requests_per_minute']}")
    print(f"Error Rate: {metrics['error_rates']['error_rate']:.2%}")
    print(f"Tokens/Second: {metrics['token_efficiency']['tokens_per_second']:.1f}")
    print(f"System Health: {health['health']} - {health['message']}")
    
    # Run load test
    print("\nStarting load test...")
    load_tester = LoadTester(None, monitor)
    load_results = await load_tester.run_load_test(
        concurrent_users=5,
        requests_per_user=10,
        test_duration=30
    )
    
    print(f"\nLoad Test Results:")
    print(f"Total Requests: {load_results['total_requests']}")
    print(f"Success Rate: {load_results['success_rate']:.2%}")
    print(f"Average Response Time: {load_results['average_response_time']:.3f}s")
    print(f"Requests/Second: {load_results['requests_per_second']:.1f}")

# Run the demo
# asyncio.run(demo_performance_monitoring())
print("Performance monitoring classes defined successfully!")
```

## 📈 Multi-Agent System Metrics

**Agent Coordination Effectiveness**:
```python
class AgentCoordinationMetrics:
    """Metrics for multi-agent system coordination and effectiveness"""
    
    def __init__(self):
        self.coordination_logs = []
        self.task_completion_logs = []
        self.communication_logs = []
    
    def log_coordination_event(self, event_data: Dict[str, Any]):
        """Log agent coordination events"""
        event_data['timestamp'] = time.time()
        self.coordination_logs.append(event_data)
    
    def evaluate_task_completion(self, task_data: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate multi-agent task completion effectiveness"""
        
        metrics = {}
        
        # Task completion rate
        total_tasks = task_data.get('total_tasks', 0)
        completed_tasks = task_data.get('completed_tasks', 0)
        metrics['completion_rate'] = completed_tasks / max(total_tasks, 1)
        
        # Average task completion time
        completion_times = task_data.get('completion_times', [])
        metrics['avg_completion_time'] = np.mean(completion_times) if completion_times else 0
        
        # Agent utilization
        agent_work_distribution = task_data.get('agent_workload', {})
        if agent_work_distribution:
            workloads = list(agent_work_distribution.values())
            metrics['workload_balance'] = 1.0 - (np.std(workloads) / max(np.mean(workloads), 1))
            metrics['agent_utilization'] = np.mean(workloads)
        
        # Communication efficiency
        comm_events = task_data.get('communication_events', 0)
        necessary_comm = task_data.get('necessary_communications', comm_events)
        metrics['communication_efficiency'] = necessary_comm / max(comm_events, 1)
        
        # Coordination overhead
        coordination_time = task_data.get('coordination_time', 0)
        total_time = task_data.get('total_execution_time', coordination_time)
        metrics['coordination_overhead'] = coordination_time / max(total_time, 1)
        
        return metrics
    
    def measure_consensus_quality(self, decisions: List[Dict[str, Any]]) -> Dict[str, float]:
        """Measure quality of multi-agent consensus decisions"""
        
        if not decisions:
            return {'consensus_score': 0.0}
        
        metrics = {}
        
        # Decision consistency
        decision_outcomes = [d.get('outcome') for d in decisions]
        unique_outcomes = len(set(decision_outcomes))
        total_decisions = len(decisions)
        metrics['decision_consistency'] = 1.0 - (unique_outcomes - 1) / max(total_decisions - 1, 1)
        
        # Confidence aggregation
        confidences = [d.get('confidence', 0.5) for d in decisions]
        metrics['avg_confidence'] = np.mean(confidences)
        metrics['confidence_variance'] = np.var(confidences)
        
        # Time to consensus
        consensus_times = [d.get('time_to_decision', 0) for d in decisions]
        metrics['avg_consensus_time'] = np.mean(consensus_times)
        
        # Participation rate
        participating_agents = set()
        for decision in decisions:
            participating_agents.update(decision.get('participating_agents', []))
        
        total_agents = len(set([agent for d in decisions 
                               for agent in d.get('all_agents', [])]))
        metrics['participation_rate'] = len(participating_agents) / max(total_agents, 1)
        
        return metrics
    
    def calculate_system_resilience(self, failure_scenarios: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate multi-agent system resilience to failures"""
        
        metrics = {}
        
        if not failure_scenarios:
            return {'resilience_score': 1.0}
        
        # Recovery time analysis
        recovery_times = []
        successful_recoveries = 0
        
        for scenario in failure_scenarios:
            if scenario.get('recovered', False):
                successful_recoveries += 1
                recovery_times.append(scenario.get('recovery_time', 0))
        
        metrics['recovery_rate'] = successful_recoveries / len(failure_scenarios)
        metrics['avg_recovery_time'] = np.mean(recovery_times) if recovery_times else float('inf')
        
        # Graceful degradation
        degradation_scores = []
        for scenario in failure_scenarios:
            failed_agents = scenario.get('failed_agents', 0)
            total_agents = scenario.get('total_agents', 1)
            performance_retained = scenario.get('performance_retained', 0.0)
            
            expected_performance = max(0, 1.0 - (failed_agents / total_agents))
            degradation_score = performance_retained / max(expected_performance, 0.1)
            degradation_scores.append(min(1.0, degradation_score))
        
        metrics['graceful_degradation'] = np.mean(degradation_scores)
        
        # Overall resilience score
        metrics['resilience_score'] = np.mean([
            metrics['recovery_rate'],
            1.0 - min(1.0, metrics['avg_recovery_time'] / 60.0),  # Normalize to 1 minute
            metrics['graceful_degradation']
        ])
        
        return metrics

# Example usage
coordination_metrics = AgentCoordinationMetrics()

# Simulate task completion data
task_data = {
    'total_tasks': 100,
    'completed_tasks': 92,
    'completion_times': np.random.uniform(5, 30, 92).tolist(),
    'agent_workload': {
        'agent_1': 25,
        'agent_2': 23,
        'agent_3': 22,
        'agent_4': 22
    },
    'communication_events': 150,
    'necessary_communications': 135,
    'coordination_time': 300,
    'total_execution_time': 1800
}

completion_metrics = coordination_metrics.evaluate_task_completion(task_data)
print("Task Completion Metrics:")
for metric, value in completion_metrics.items():
    print(f"  {metric}: {value:.3f}")

# Simulate consensus decisions
decisions = [
    {
        'outcome': 'approve',
        'confidence': 0.85,
        'time_to_decision': 15,
        'participating_agents': ['agent_1', 'agent_2', 'agent_3'],
        'all_agents': ['agent_1', 'agent_2', 'agent_3', 'agent_4']
    },
    {
        'outcome': 'approve',
        'confidence': 0.78,
        'time_to_decision': 22,
        'participating_agents': ['agent_1', 'agent_2', 'agent_4'],
        'all_agents': ['agent_1', 'agent_2', 'agent_3', 'agent_4']
    }
]

consensus_metrics = coordination_metrics.measure_consensus_quality(decisions)
print("\nConsensus Quality Metrics:")
for metric, value in consensus_metrics.items():
    print(f"  {metric}: {value:.3f}")
```

## 🔍 Real-time Monitoring Dashboard

**Monitoring System Implementation**:
```python
import json
from datetime import datetime, timedelta
import sqlite3

class RealTimeMonitor:
    """Real-time monitoring system for LLM applications"""
    
    def __init__(self, db_path: str = "monitoring.db"):
        self.db_path = db_path
        self.init_database()
        self.alert_thresholds = {
            'response_time_p95': 5.0,  # seconds
            'error_rate': 0.05,        # 5%
            'memory_usage': 0.8,       # 80%
            'cpu_usage': 0.9           # 90%
        }
    
    def init_database(self):
        """Initialize monitoring database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                metric_type TEXT,
                metric_name TEXT,
                value REAL,
                labels TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                alert_type TEXT,
                message TEXT,
                severity TEXT,
                resolved BOOLEAN DEFAULT FALSE
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def record_metric(self, metric_type: str, metric_name: str, 
                     value: float, labels: Dict[str, str] = None):
        """Record a metric value"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO metrics (timestamp, metric_type, metric_name, value, labels)
            VALUES (?, ?, ?, ?, ?)
        ''', (time.time(), metric_type, metric_name, value, 
              json.dumps(labels or {})))
        
        conn.commit()
        conn.close()
        
        # Check for alerts
        self._check_alerts(metric_type, metric_name, value)
    
    def _check_alerts(self, metric_type: str, metric_name: str, value: float):
        """Check if metric value triggers alerts"""
        threshold_key = f"{metric_name}"
        
        if threshold_key in self.alert_thresholds:
            threshold = self.alert_thresholds[threshold_key]
            
            if value > threshold:
                self._create_alert(
                    alert_type="threshold_exceeded",
                    message=f"{metric_name} exceeded threshold: {value:.3f} > {threshold}",
                    severity="warning" if value < threshold * 1.2 else "critical"
                )
    
    def _create_alert(self, alert_type: str, message: str, severity: str):
        """Create an alert"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO alerts (timestamp, alert_type, message, severity)
            VALUES (?, ?, ?, ?)
        ''', (time.time(), alert_type, message, severity))
        
        conn.commit()
        conn.close()
        
        print(f"ALERT [{severity.upper()}]: {message}")
    
    def get_metrics_summary(self, time_window_minutes: int = 60) -> Dict[str, Any]:
        """Get metrics summary for specified time window"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cutoff_time = time.time() - (time_window_minutes * 60)
        
        cursor.execute('''
            SELECT metric_type, metric_name, AVG(value) as avg_value,
                   MIN(value) as min_value, MAX(value) as max_value,
                   COUNT(*) as count
            FROM metrics 
            WHERE timestamp > ?
            GROUP BY metric_type, metric_name
        ''', (cutoff_time,))
        
        results = cursor.fetchall()
        conn.close()
        
        summary = {}
        for row in results:
            metric_type, metric_name, avg_val, min_val, max_val, count = row
            
            if metric_type not in summary:
                summary[metric_type] = {}
            
            summary[metric_type][metric_name] = {
                'average': avg_val,
                'minimum': min_val,
                'maximum': max_val,
                'sample_count': count
            }
        
        return summary
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get active alerts"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, timestamp, alert_type, message, severity
            FROM alerts 
            WHERE resolved = FALSE
            ORDER BY timestamp DESC
        ''')
        
        results = cursor.fetchall()
        conn.close()
        
        alerts = []
        for row in results:
            alert_id, timestamp, alert_type, message, severity = row
            alerts.append({
                'id': alert_id,
                'timestamp': datetime.fromtimestamp(timestamp).isoformat(),
                'type': alert_type,
                'message': message,
                'severity': severity
            })
        
        return alerts
    
    def generate_dashboard_data(self) -> Dict[str, Any]:
        """Generate data for monitoring dashboard"""
        return {
            'timestamp': datetime.now().isoformat(),
            'metrics_summary': self.get_metrics_summary(60),
            'active_alerts': self.get_active_alerts(),
            'system_status': self._get_system_status()
        }
    
    def _get_system_status(self) -> Dict[str, str]:
        """Get overall system status"""
        active_alerts = self.get_active_alerts()
        
        critical_alerts = [a for a in active_alerts if a['severity'] == 'critical']
        warning_alerts = [a for a in active_alerts if a['severity'] == 'warning']
        
        if critical_alerts:
            return {
                'status': 'critical',
                'message': f"{len(critical_alerts)} critical alerts active"
            }
        elif warning_alerts:
            return {
                'status': 'warning', 
                'message': f"{len(warning_alerts)} warnings active"
            }
        else:
            return {
                'status': 'healthy',
                'message': 'All systems operational'
            }

# Example usage and integration
monitor = RealTimeMonitor()

# Simulate metrics collection
for i in range(100):
    # Response time metrics
    response_time = np.random.uniform(0.5, 4.0)
    if i > 50:  # Simulate degradation
        response_time *= 2
    
    monitor.record_metric('performance', 'response_time_p95', response_time)
    
    # Error rate metrics  
    error_rate = np.random.uniform(0.01, 0.03)
    if i > 80:  # Simulate errors
        error_rate = 0.08
    
    monitor.record_metric('reliability', 'error_rate', error_rate)
    
    time.sleep(0.1)

# Generate dashboard
dashboard_data = monitor.generate_dashboard_data()
print("Monitoring Dashboard Data:")
print(json.dumps(dashboard_data, indent=2))
```

## ✅ Performance Measurement Checklist

**Quality Metrics**:
- [ ] BLEU/ROUGE scores for text generation
- [ ] Factual accuracy verification
- [ ] Coherence and fluency scoring
- [ ] Bias detection and measurement
- [ ] Safety and appropriateness scoring

**Efficiency Metrics**:
- [ ] Response time percentiles (P50, P95, P99)
- [ ] Throughput (requests/second)
- [ ] Token processing efficiency
- [ ] Resource utilization monitoring
- [ ] Cost per request/token

**System Metrics**:
- [ ] Error rates and types
- [ ] Availability and uptime
- [ ] Load balancing effectiveness
- [ ] Auto-scaling responsiveness
- [ ] Disaster recovery capability

**Multi-Agent Metrics**:
- [ ] Task completion rates
- [ ] Agent coordination efficiency
- [ ] Consensus quality measurement
- [ ] System resilience to failures
- [ ] Communication overhead tracking

## 🚀 Next Steps

Continue with:

1. **[Benchmarking](benchmarking.md)** - Comprehensive benchmark suites
2. **[Monitoring Systems](monitoring.md)** - Production monitoring setup
3. **[Optimization Techniques](optimization.md)** - Performance optimization strategies

---

*Comprehensive performance measurement is essential for building reliable, efficient, and high-quality LLM systems. Use these metrics to continuously improve your AI applications.*
