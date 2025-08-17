# Prompt Engineering: Effective LLM Communication

Prompt engineering is the art and science of crafting inputs that guide LLMs to produce desired outputs. This section covers advanced techniques for getting the best performance from language models.

## 🎯 Core Prompt Engineering Principles

### Understanding Prompt Structure

**Anatomy of an Effective Prompt**:
```python
class PromptTemplate:
    """Structured approach to prompt creation"""
    
    def __init__(self):
        self.components = {
            'context': '',
            'role': '',
            'task': '',
            'format': '',
            'examples': [],
            'constraints': []
        }
    
    def build_prompt(self, **kwargs) -> str:
        """Build structured prompt from components"""
        prompt_parts = []
        
        # Role definition
        if self.components['role']:
            prompt_parts.append(f"You are {self.components['role']}.")
        
        # Context setting
        if self.components['context']:
            prompt_parts.append(f"Context: {self.components['context']}")
        
        # Task specification
        if self.components['task']:
            prompt_parts.append(f"Task: {self.components['task']}")
        
        # Examples (few-shot)
        if self.components['examples']:
            prompt_parts.append("Examples:")
            for i, example in enumerate(self.components['examples'], 1):
                prompt_parts.append(f"{i}. {example}")
        
        # Format specification
        if self.components['format']:
            prompt_parts.append(f"Format: {self.components['format']}")
        
        # Constraints
        if self.components['constraints']:
            prompt_parts.append("Constraints:")
            for constraint in self.components['constraints']:
                prompt_parts.append(f"- {constraint}")
        
        # Insert dynamic content
        prompt = '\n\n'.join(prompt_parts)
        for key, value in kwargs.items():
            prompt = prompt.replace(f'{{{key}}}', str(value))
        
        return prompt
    
    def set_role(self, role: str):
        """Set the role/persona for the assistant"""
        self.components['role'] = role
        return self
    
    def set_context(self, context: str):
        """Set background context"""
        self.components['context'] = context
        return self
    
    def set_task(self, task: str):
        """Define the specific task"""
        self.components['task'] = task
        return self
    
    def set_format(self, format_spec: str):
        """Specify output format"""
        self.components['format'] = format_spec
        return self
    
    def add_example(self, example: str):
        """Add few-shot example"""
        self.components['examples'].append(example)
        return self
    
    def add_constraint(self, constraint: str):
        """Add constraint or requirement"""
        self.components['constraints'].append(constraint)
        return self

# Example usage
code_review_template = PromptTemplate()
code_review_template.set_role("an experienced software engineer and code reviewer")
code_review_template.set_context("You are reviewing code for a production system")
code_review_template.set_task("Review the following code and provide feedback on {focus_areas}")
code_review_template.set_format("Provide feedback in the format: [ISSUE/SUGGESTION]: Description")
code_review_template.add_constraint("Focus on security, performance, and maintainability")
code_review_template.add_constraint("Be specific and actionable")
code_review_template.add_example("Input: def unsafe_query(user_input): return f'SELECT * FROM users WHERE name = {user_input}'\nOutput: [SECURITY ISSUE]: SQL injection vulnerability - use parameterized queries")

prompt = code_review_template.build_prompt(focus_areas="security and performance")
print("Generated Prompt:")
print(prompt)
```

### Chain of Thought (CoT) Prompting

**Step-by-Step Reasoning**:
```python
class ChainOfThoughtPrompt:
    """Generate Chain of Thought prompts for complex reasoning"""
    
    def __init__(self):
        self.reasoning_patterns = {
            'mathematical': [
                "Let me work through this step by step:",
                "First, I need to identify what we're looking for:",
                "Then, I'll set up the equation:",
                "Now I'll solve step by step:",
                "Finally, I'll check my answer:"
            ],
            'analytical': [
                "Let me analyze this systematically:",
                "First, I'll gather the key information:",
                "Next, I'll consider different perspectives:",
                "Then, I'll evaluate the evidence:",
                "Finally, I'll draw a conclusion:"
            ],
            'creative': [
                "Let me approach this creatively:",
                "First, I'll brainstorm ideas:",
                "Then, I'll develop the most promising concepts:",
                "Next, I'll add details and refinements:",
                "Finally, I'll present the final result:"
            ]
        }
    
    def create_cot_prompt(self, task: str, reasoning_type: str = 'analytical') -> str:
        """Create Chain of Thought prompt"""
        if reasoning_type not in self.reasoning_patterns:
            reasoning_type = 'analytical'
        
        steps = self.reasoning_patterns[reasoning_type]
        
        prompt = f"""Task: {task}

Please solve this step-by-step using the following approach:

{chr(10).join(f"{i+1}. {step}" for i, step in enumerate(steps))}

Think through each step carefully and show your reasoning process."""
        
        return prompt
    
    def create_few_shot_cot(self, task: str, examples: list) -> str:
        """Create few-shot Chain of Thought prompt with examples"""
        prompt_parts = []
        
        # Add examples
        for i, example in enumerate(examples, 1):
            prompt_parts.append(f"Example {i}:")
            prompt_parts.append(f"Question: {example['question']}")
            prompt_parts.append(f"Reasoning: {example['reasoning']}")
            prompt_parts.append(f"Answer: {example['answer']}")
            prompt_parts.append("")
        
        # Add the actual task
        prompt_parts.append("Now solve this problem:")
        prompt_parts.append(f"Question: {task}")
        prompt_parts.append("Reasoning: Let me think step by step.")
        
        return "\n".join(prompt_parts)

# Mathematical reasoning example
cot_prompt = ChainOfThoughtPrompt()

math_examples = [
    {
        'question': 'A store has 23 apples and sells 8. How many are left?',
        'reasoning': 'I need to subtract the sold apples from the total. 23 - 8 = 15',
        'answer': '15 apples'
    }
]

math_prompt = cot_prompt.create_few_shot_cot(
    "A bakery makes 144 cookies and sells them in boxes of 12. How many boxes can they make?",
    math_examples
)

print("Chain of Thought Math Prompt:")
print(math_prompt)
```

## 🔧 Advanced Prompting Techniques

### Zero-Shot, Few-Shot, and Many-Shot Learning

**Adaptive Shot Selection**:
```python
class AdaptiveShotLearning:
    """Dynamically choose between zero-shot, few-shot, and many-shot prompting"""
    
    def __init__(self):
        self.performance_cache = {}
        self.example_database = {}
    
    def add_examples(self, task_type: str, examples: list):
        """Add examples for a specific task type"""
        self.example_database[task_type] = examples
    
    def zero_shot_prompt(self, task: str, instruction: str) -> str:
        """Create zero-shot prompt"""
        return f"""Instruction: {instruction}

Task: {task}

Please complete this task based on the instruction above."""
    
    def few_shot_prompt(self, task: str, task_type: str, num_examples: int = 3) -> str:
        """Create few-shot prompt with examples"""
        if task_type not in self.example_database:
            return self.zero_shot_prompt(task, f"Complete this {task_type} task")
        
        examples = self.example_database[task_type][:num_examples]
        
        prompt_parts = ["Here are some examples of this task:"]
        
        for i, example in enumerate(examples, 1):
            prompt_parts.append(f"\nExample {i}:")
            prompt_parts.append(f"Input: {example['input']}")
            prompt_parts.append(f"Output: {example['output']}")
        
        prompt_parts.append(f"\nNow complete this task:")
        prompt_parts.append(f"Input: {task}")
        prompt_parts.append("Output:")
        
        return "\n".join(prompt_parts)
    
    def many_shot_prompt(self, task: str, task_type: str) -> str:
        """Create many-shot prompt with comprehensive examples"""
        if task_type not in self.example_database:
            return self.few_shot_prompt(task, task_type)
        
        examples = self.example_database[task_type]
        
        prompt_parts = [f"You will complete a {task_type} task. Here are many examples to learn from:"]
        
        for i, example in enumerate(examples, 1):
            prompt_parts.append(f"\nExample {i}:")
            prompt_parts.append(f"Input: {example['input']}")
            prompt_parts.append(f"Output: {example['output']}")
        
        prompt_parts.append("\nBased on these examples, complete the following:")
        prompt_parts.append(f"Input: {task}")
        prompt_parts.append("Output:")
        
        return "\n".join(prompt_parts)
    
    def select_best_approach(self, task: str, task_type: str, complexity: str = 'medium') -> str:
        """Automatically select the best prompting approach"""
        # Simple heuristics for approach selection
        if complexity == 'simple' or task_type not in self.example_database:
            return self.zero_shot_prompt(task, f"Complete this {task_type} task")
        elif complexity == 'medium':
            return self.few_shot_prompt(task, task_type)
        else:  # complex
            return self.many_shot_prompt(task, task_type)

# Example: Sentiment analysis
shot_learner = AdaptiveShotLearning()

# Add sentiment analysis examples
sentiment_examples = [
    {'input': 'I love this product!', 'output': 'Positive'},
    {'input': 'This is terrible.', 'output': 'Negative'},
    {'input': 'It\'s okay, I guess.', 'output': 'Neutral'},
    {'input': 'Amazing quality and fast delivery!', 'output': 'Positive'},
    {'input': 'Worst purchase ever.', 'output': 'Negative'},
]

shot_learner.add_examples('sentiment_analysis', sentiment_examples)

# Generate different prompts
task = "The movie was decent but too long."

zero_shot = shot_learner.zero_shot_prompt(task, "Analyze the sentiment of this text")
few_shot = shot_learner.few_shot_prompt(task, 'sentiment_analysis')
auto_selected = shot_learner.select_best_approach(task, 'sentiment_analysis', 'medium')

print("Zero-shot prompt:")
print(zero_shot)
print("\n" + "="*50 + "\n")
print("Few-shot prompt:")
print(few_shot)
```

### Role-Based Prompting and Personas

**Expert Persona Creation**:
```python
class ExpertPersona:
    """Create expert personas for specialized tasks"""
    
    def __init__(self):
        self.personas = {
            'software_architect': {
                'background': 'Senior Software Architect with 15 years experience in distributed systems',
                'expertise': ['system design', 'scalability', 'microservices', 'cloud architecture'],
                'approach': 'methodical and thorough',
                'communication_style': 'technical but clear'
            },
            'data_scientist': {
                'background': 'Senior Data Scientist with PhD in Statistics and 10 years industry experience',
                'expertise': ['machine learning', 'statistical analysis', 'data visualization', 'predictive modeling'],
                'approach': 'evidence-based and analytical',
                'communication_style': 'precise with proper statistical terminology'
            },
            'creative_writer': {
                'background': 'Award-winning author and creative writing professor',
                'expertise': ['storytelling', 'character development', 'narrative structure', 'literary analysis'],
                'approach': 'imaginative and expressive',
                'communication_style': 'engaging and vivid'
            },
            'business_consultant': {
                'background': 'Management Consultant with MBA and 12 years at top consulting firms',
                'expertise': ['strategy', 'process improvement', 'change management', 'financial analysis'],
                'approach': 'strategic and results-oriented',
                'communication_style': 'professional and actionable'
            }
        }
    
    def create_persona_prompt(self, persona_type: str, task: str, additional_context: str = '') -> str:
        """Create a prompt with expert persona"""
        if persona_type not in self.personas:
            return f"As an expert, please help with: {task}"
        
        persona = self.personas[persona_type]
        
        prompt_parts = [
            f"You are a {persona['background']}.",
            f"Your expertise includes: {', '.join(persona['expertise'])}.",
            f"Your approach is {persona['approach']}.",
            f"Please communicate in a {persona['communication_style']} manner.",
        ]
        
        if additional_context:
            prompt_parts.append(f"Additional context: {additional_context}")
        
        prompt_parts.extend([
            f"",
            f"Task: {task}",
            f"",
            f"Please provide your expert analysis and recommendations:"
        ])
        
        return "\n".join(prompt_parts)
    
    def create_multi_expert_prompt(self, task: str, persona_types: list) -> str:
        """Create prompt with multiple expert perspectives"""
        prompt_parts = [
            f"I need multiple expert perspectives on the following task: {task}",
            "",
            "Please provide analysis from the following expert viewpoints:",
            ""
        ]
        
        for i, persona_type in enumerate(persona_types, 1):
            if persona_type in self.personas:
                persona = self.personas[persona_type]
                prompt_parts.append(f"{i}. {persona['background']}:")
                prompt_parts.append(f"   Focus on: {', '.join(persona['expertise'][:2])}")
                prompt_parts.append("")
        
        prompt_parts.append("Provide distinct insights from each expert perspective, highlighting areas of agreement and disagreement.")
        
        return "\n".join(prompt_parts)

# Example usage
persona_creator = ExpertPersona()

# Single expert prompt
arch_prompt = persona_creator.create_persona_prompt(
    'software_architect',
    'Design a microservices architecture for a high-traffic e-commerce platform',
    'Expected 100k+ concurrent users with global distribution'
)

print("Software Architect Persona Prompt:")
print(arch_prompt)
print("\n" + "="*50 + "\n")

# Multi-expert prompt
multi_expert = persona_creator.create_multi_expert_prompt(
    'Should our startup adopt a microservices architecture?',
    ['software_architect', 'business_consultant']
)

print("Multi-Expert Prompt:")
print(multi_expert)
```

## 🧠 Cognitive Prompting Patterns

### Self-Correction and Validation

**Self-Reflective Prompting**:
```python
class SelfCorrectivePrompt:
    """Implement self-correction and validation patterns"""
    
    def __init__(self):
        self.validation_patterns = {
            'mathematical': [
                "Check: Does this answer make sense given the problem?",
                "Verify: Can I work backwards to confirm?",
                "Test: What happens if I substitute this answer?"
            ],
            'logical': [
                "Check: Are my premises correct?",
                "Verify: Does my conclusion follow logically?",
                "Test: Can I find any counterexamples?"
            ],
            'factual': [
                "Check: Am I certain about these facts?",
                "Verify: Are there any contradictions?",
                "Test: What sources support this information?"
            ]
        }
    
    def create_self_corrective_prompt(self, task: str, domain: str = 'logical') -> str:
        """Create prompt with built-in self-correction"""
        validation_steps = self.validation_patterns.get(domain, self.validation_patterns['logical'])
        
        prompt = f"""Task: {task}

Please complete this task using the following process:

1. **Initial Response**: Provide your first answer or solution.

2. **Self-Review**: Now critically examine your response:
   {chr(10).join(f"   - {step}" for step in validation_steps)}

3. **Correction**: If you found any issues in step 2, provide a corrected response. If not, reaffirm your initial answer.

4. **Final Answer**: State your final, verified response.

Begin with your initial response:"""
        
        return prompt
    
    def create_multi_perspective_validation(self, task: str) -> str:
        """Create prompt that validates from multiple angles"""
        return f"""Task: {task}

Please approach this from multiple angles to ensure accuracy:

**Step 1 - Initial Analysis**: Provide your first response.

**Step 2 - Alternative Approach**: Solve or analyze this using a different method or perspective.

**Step 3 - Devil's Advocate**: What arguments could be made against your conclusion?

**Step 4 - Edge Cases**: What edge cases or exceptions should be considered?

**Step 5 - Synthesis**: Combine insights from all perspectives into a final, well-validated response.

Begin with Step 1:"""

# Example usage
corrective_prompt = SelfCorrectivePrompt()

math_prompt = corrective_prompt.create_self_corrective_prompt(
    "If a train travels 120 miles in 2 hours, then speeds up and travels the next 180 miles in 1.5 hours, what is the average speed for the entire journey?",
    domain='mathematical'
)

print("Self-Corrective Math Prompt:")
print(math_prompt)
```

### Tree of Thoughts (ToT) Prompting

**Branching Reasoning Patterns**:
```python
class TreeOfThoughtsPrompt:
    """Implement Tree of Thoughts prompting for complex problem solving"""
    
    def __init__(self):
        self.thought_evaluation_criteria = [
            "Feasibility: How realistic is this approach?",
            "Creativity: How novel or innovative is this idea?",
            "Effectiveness: How well would this solve the problem?",
            "Efficiency: How resource-friendly is this solution?"
        ]
    
    def create_tot_prompt(self, problem: str, num_thoughts: int = 3, depth: int = 2) -> str:
        """Create Tree of Thoughts prompt"""
        prompt_parts = [
            f"Problem: {problem}",
            "",
            "I will solve this using a Tree of Thoughts approach:",
            "",
            f"**Level 1: Generate {num_thoughts} different approaches**"
        ]
        
        for i in range(num_thoughts):
            prompt_parts.append(f"Thought {i+1}: [Generate a distinct approach or solution path]")
        
        prompt_parts.extend([
            "",
            "**Level 1 Evaluation**: For each thought, evaluate based on:",
        ])
        
        for criterion in self.thought_evaluation_criteria:
            prompt_parts.append(f"- {criterion}")
        
        if depth > 1:
            prompt_parts.extend([
                "",
                "**Level 2: Expand the most promising thought(s)**",
                "Select the best 1-2 thoughts from Level 1 and generate sub-approaches:",
                "",
                "Expanded Thought A: [Develop the selected approach further]",
                "Expanded Thought B: [Develop alternative refinements]",
                "",
                "**Level 2 Evaluation**: Compare the expanded thoughts and select the best path.",
            ])
        
        prompt_parts.extend([
            "",
            "**Final Solution**: Based on the tree exploration above, provide the optimal solution with reasoning for why this path was chosen.",
            "",
            "Let's begin with Level 1:"
        ])
        
        return "\n".join(prompt_parts)
    
    def create_collaborative_tot(self, problem: str) -> str:
        """Create collaborative Tree of Thoughts with multiple perspectives"""
        return f"""Problem: {problem}

I will explore this problem using multiple thinking modes:

**🔍 Analytical Mode**: Break down the problem systematically
- Thought 1: [Logical, step-by-step approach]
- Evaluation: [Assess logical soundness]

**💡 Creative Mode**: Generate innovative solutions  
- Thought 2: [Novel, out-of-the-box approach]
- Evaluation: [Assess creativity and uniqueness]

**⚡ Practical Mode**: Focus on implementable solutions
- Thought 3: [Pragmatic, resource-conscious approach]  
- Evaluation: [Assess feasibility and efficiency]

**🌟 Synthesis**: Combine the best elements from each mode into a unified solution.

**🎯 Final Recommendation**: Present the optimal approach with clear reasoning.

Let's start with Analytical Mode:"""

# Example usage
tot_prompt = TreeOfThoughtsPrompt()

complex_problem = "Design a system to reduce food waste in restaurants while maintaining food quality and profitability."

tot_solution_prompt = tot_prompt.create_tot_prompt(complex_problem, num_thoughts=3, depth=2)

print("Tree of Thoughts Prompt:")
print(tot_solution_prompt)
```

## 📊 Prompt Optimization and Testing

### A/B Testing for Prompts

**Systematic Prompt Evaluation**:
```python
import random
from typing import Dict, List, Tuple
from collections import defaultdict

class PromptOptimizer:
    """Optimize prompts through systematic testing and evaluation"""
    
    def __init__(self):
        self.test_results = defaultdict(list)
        self.prompt_variants = {}
        self.evaluation_metrics = [
            'accuracy', 'relevance', 'clarity', 'completeness', 'efficiency'
        ]
    
    def add_prompt_variant(self, variant_name: str, prompt_template: str):
        """Add a prompt variant for testing"""
        self.prompt_variants[variant_name] = prompt_template
    
    def evaluate_response(self, response: str, expected_criteria: Dict[str, str]) -> Dict[str, float]:
        """Evaluate a response against criteria (simplified simulation)"""
        # In practice, this would use actual evaluation methods
        # This is a simplified simulation
        
        scores = {}
        
        # Simulate scoring based on response characteristics
        word_count = len(response.split())
        
        scores['accuracy'] = min(1.0, len([word for word in expected_criteria.get('keywords', []) 
                                         if word.lower() in response.lower()]) / max(1, len(expected_criteria.get('keywords', []))))
        
        scores['relevance'] = 0.8 + 0.2 * random.random()  # Simulated
        scores['clarity'] = max(0.3, 1.0 - abs(word_count - 100) / 200)  # Prefer ~100 words
        scores['completeness'] = min(1.0, word_count / max(1, expected_criteria.get('min_words', 50)))
        scores['efficiency'] = max(0.5, 1.2 - word_count / 200)  # Prefer conciseness
        
        return scores
    
    def run_ab_test(self, test_cases: List[Dict], num_iterations: int = 10) -> Dict[str, Dict[str, float]]:
        """Run A/B test across prompt variants"""
        results = defaultdict(lambda: defaultdict(list))
        
        print(f"Running A/B test with {len(self.prompt_variants)} variants on {len(test_cases)} test cases...")
        
        for iteration in range(num_iterations):
            for test_case in test_cases:
                task = test_case['task']
                criteria = test_case['criteria']
                
                for variant_name, prompt_template in self.prompt_variants.items():
                    # Generate full prompt
                    full_prompt = prompt_template.format(**test_case.get('variables', {}))
                    
                    # Simulate response generation (in practice, would call LLM)
                    response = self.simulate_llm_response(full_prompt, task)
                    
                    # Evaluate response
                    scores = self.evaluate_response(response, criteria)
                    
                    # Store results
                    for metric, score in scores.items():
                        results[variant_name][metric].append(score)
        
        # Calculate average scores
        final_results = {}
        for variant_name in results:
            final_results[variant_name] = {
                metric: sum(scores) / len(scores) 
                for metric, scores in results[variant_name].items()
            }
        
        return final_results
    
    def simulate_llm_response(self, prompt: str, task: str) -> str:
        """Simulate LLM response (placeholder)"""
        # This would be replaced with actual LLM API calls
        response_templates = [
            f"Based on the prompt, here's my analysis of {task}: This is a comprehensive response that addresses the key points.",
            f"To solve {task}, I need to consider multiple factors and provide a detailed solution.",
            f"The task of {task} requires careful consideration. Here's my approach and recommendations."
        ]
        
        # Vary response length based on prompt characteristics
        prompt_length = len(prompt.split())
        if prompt_length > 100:
            return response_templates[1] + " " + "Additional detailed analysis follows with specific examples and reasoning."
        elif "concise" in prompt.lower():
            return response_templates[0]
        else:
            return response_templates[2]
    
    def generate_optimization_report(self, results: Dict[str, Dict[str, float]]) -> str:
        """Generate optimization report"""
        report_parts = ["# Prompt Optimization Report\n"]
        
        # Overall winner
        overall_scores = {
            variant: sum(scores.values()) / len(scores.values())
            for variant, scores in results.items()
        }
        
        best_variant = max(overall_scores.keys(), key=lambda k: overall_scores[k])
        
        report_parts.append(f"## 🏆 Best Overall Performer: {best_variant}")
        report_parts.append(f"Average Score: {overall_scores[best_variant]:.3f}\n")
        
        # Detailed breakdown
        report_parts.append("## 📊 Detailed Results\n")
        
        for variant_name, scores in results.items():
            report_parts.append(f"### {variant_name}")
            for metric, score in scores.items():
                report_parts.append(f"- {metric.title()}: {score:.3f}")
            report_parts.append(f"- **Overall**: {overall_scores[variant_name]:.3f}\n")
        
        # Recommendations
        report_parts.append("## 💡 Recommendations\n")
        
        # Find best performer for each metric
        for metric in self.evaluation_metrics:
            best_for_metric = max(results.keys(), key=lambda k: results[k].get(metric, 0))
            report_parts.append(f"- **Best for {metric}**: {best_for_metric} ({results[best_for_metric][metric]:.3f})")
        
        return "\n".join(report_parts)

# Example optimization test
optimizer = PromptOptimizer()

# Add different prompt variants
optimizer.add_prompt_variant(
    'direct', 
    "Task: {task}\n\nPlease provide a direct answer to this task."
)

optimizer.add_prompt_variant(
    'structured',
    "Task: {task}\n\nPlease approach this systematically:\n1. Analysis\n2. Solution\n3. Conclusion"
)

optimizer.add_prompt_variant(
    'expert_persona',
    "You are an expert in {domain}. Task: {task}\n\nProvide your expert analysis and recommendations."
)

optimizer.add_prompt_variant(
    'step_by_step',
    "Task: {task}\n\nLet me work through this step by step:\n\nStep 1: Understand the problem\nStep 2: Analyze options\nStep 3: Provide solution"
)

# Define test cases
test_cases = [
    {
        'task': 'analyze market trends for electric vehicles',
        'domain': 'automotive industry analysis',
        'variables': {'domain': 'automotive industry analysis'},
        'criteria': {
            'keywords': ['market', 'trends', 'electric', 'vehicles', 'analysis'],
            'min_words': 50
        }
    },
    {
        'task': 'design a database schema for an e-commerce platform',
        'domain': 'database design',
        'variables': {'domain': 'database design'},
        'criteria': {
            'keywords': ['database', 'schema', 'e-commerce', 'design', 'tables'],
            'min_words': 75
        }
    }
]

# Run optimization
results = optimizer.run_ab_test(test_cases, num_iterations=5)
report = optimizer.generate_optimization_report(results)

print(report)
```

### Dynamic Prompt Adaptation

**Context-Aware Prompt Selection**:
```python
class AdaptivePromptSystem:
    """Dynamically adapt prompts based on context and performance"""
    
    def __init__(self):
        self.context_patterns = {
            'technical': {
                'keywords': ['api', 'database', 'algorithm', 'code', 'system'],
                'preferred_style': 'structured',
                'max_length': 500
            },
            'creative': {
                'keywords': ['story', 'design', 'creative', 'artistic', 'innovative'],
                'preferred_style': 'open_ended',
                'max_length': 300
            },
            'analytical': {
                'keywords': ['analyze', 'compare', 'evaluate', 'assess', 'examine'],
                'preferred_style': 'methodical',
                'max_length': 400
            },
            'explanatory': {
                'keywords': ['explain', 'describe', 'clarify', 'define', 'illustrate'],
                'preferred_style': 'educational',
                'max_length': 350
            }
        }
        
        self.prompt_styles = {
            'structured': "Please approach this systematically with clear sections and bullet points.",
            'open_ended': "Feel free to be creative and explore different possibilities.",
            'methodical': "Use a step-by-step analytical approach with evidence and reasoning.",
            'educational': "Explain clearly as if teaching someone new to this topic."
        }
        
        self.performance_history = defaultdict(list)
    
    def detect_context(self, task: str) -> str:
        """Detect the context type of a task"""
        task_lower = task.lower()
        
        context_scores = {}
        for context_type, patterns in self.context_patterns.items():
            score = sum(1 for keyword in patterns['keywords'] if keyword in task_lower)
            context_scores[context_type] = score
        
        # Return context with highest score, default to analytical
        if max(context_scores.values()) == 0:
            return 'analytical'
        
        return max(context_scores.keys(), key=context_scores.get)
    
    def generate_adaptive_prompt(self, task: str, user_preferences: Dict = None) -> str:
        """Generate context-appropriate prompt"""
        context = self.detect_context(task)
        context_config = self.context_patterns[context]
        
        # Build adaptive prompt
        prompt_parts = []
        
        # Add context-specific guidance
        style_instruction = self.prompt_styles[context_config['preferred_style']]
        prompt_parts.append(style_instruction)
        
        # Add length guidance
        max_length = context_config['max_length']
        prompt_parts.append(f"Please keep your response to approximately {max_length} words.")
        
        # User preferences override
        if user_preferences:
            if 'style' in user_preferences:
                custom_style = self.prompt_styles.get(user_preferences['style'], style_instruction)
                prompt_parts[0] = custom_style
            
            if 'length' in user_preferences:
                prompt_parts[1] = f"Please keep your response to approximately {user_preferences['length']} words."
        
        # Add the actual task
        prompt_parts.extend([
            "",
            f"Task: {task}",
            "",
            "Your response:"
        ])
        
        return "\n".join(prompt_parts)
    
    def update_performance(self, prompt_type: str, task: str, performance_score: float):
        """Update performance history for learning"""
        self.performance_history[prompt_type].append({
            'task': task,
            'score': performance_score,
            'timestamp': 'now'  # In practice, use actual timestamp
        })
    
    def get_best_approach(self, similar_task: str) -> str:
        """Recommend best approach based on historical performance"""
        if not self.performance_history:
            return self.detect_context(similar_task)
        
        # Calculate average performance for each approach
        avg_performance = {}
        for approach, history in self.performance_history.items():
            if history:
                avg_performance[approach] = sum(h['score'] for h in history) / len(history)
        
        if not avg_performance:
            return self.detect_context(similar_task)
        
        # Return approach with best average performance
        best_approach = max(avg_performance.keys(), key=avg_performance.get)
        return best_approach

# Example usage
adaptive_system = AdaptivePromptSystem()

# Test different tasks
tasks = [
    "Explain how machine learning algorithms work",
    "Design a creative marketing campaign for eco-friendly products", 
    "Analyze the performance metrics of our web application",
    "Write a Python function to process user data"
]

print("Adaptive Prompt Examples:")
print("=" * 50)

for task in tasks:
    context = adaptive_system.detect_context(task)
    adaptive_prompt = adaptive_system.generate_adaptive_prompt(task)
    
    print(f"Task: {task}")
    print(f"Detected Context: {context}")
    print(f"Adaptive Prompt:")
    print(adaptive_prompt)
    print("-" * 30)
```

## ✅ Prompt Engineering Mastery Checklist

Ensure you can:

1. **Structure Effective Prompts**: Role, context, task, format, constraints
2. **Apply CoT Reasoning**: Step-by-step problem solving
3. **Use Few-Shot Learning**: Provide relevant examples
4. **Create Expert Personas**: Role-based prompting
5. **Implement Self-Correction**: Built-in validation patterns
6. **Optimize Through Testing**: A/B testing and metrics
7. **Adapt Dynamically**: Context-aware prompt selection

## 🚀 Next Steps

With prompt engineering skills mastered, continue to:

1. **[Model Evaluation](evaluation.md)** - Assess LLM performance systematically
2. **[Building LLM Agents](../agents/architecture.md)** - Apply prompting to agent systems
3. **[Multi-Agent Platforms](../mcp/fundamentals.md)** - Coordinate multiple LLM agents

---

*Prompt engineering is a critical skill for working effectively with LLMs in agent systems. Master these techniques to get optimal performance from your language models and build more reliable AI applications.*
