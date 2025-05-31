import torch
import json
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import numpy as np
import re
import warnings
warnings.filterwarnings('ignore')

class MedicalBenchmarkEvaluator:
    def __init__(self, model_path):
        print("🔄 Loading model for benchmark evaluation...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        # Set pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("✅ Model loaded successfully!")
    
    def generate_response(self, prompt, max_new_tokens=50):
        """Generate response from model - FIXED VERSION"""
        # Truncate input to reasonable length
        inputs = self.tokenizer(
            prompt, 
            return_tensors='pt', 
            truncation=True, 
            max_length=300  # Limit input length
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=max_new_tokens,  # Only use max_new_tokens
                temperature=0.3,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode only the new tokens (skip the input)
        input_length = inputs['input_ids'].shape[1]
        new_tokens = outputs[0][input_length:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return response.strip()
    
    def extract_answer_choice(self, response, choices=['A', 'B', 'C', 'D', 'E']):
        """Extract answer choice from response"""
        response_upper = response.upper()
        
        # Look for explicit answer patterns
        patterns = [
            r'\b([A-E])\b',
            r'\(([A-E])\)',
            r'([A-E])[\)\.\:]'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, response_upper)
            if matches:
                answer = matches[0] if isinstance(matches[0], str) else matches[0][-1]
                if answer in choices:
                    return answer
        
        return None
    
    def test_pubmedqa(self, num_samples=10):
        """Test on PubMedQA benchmark - SIMPLIFIED"""
        print(f"📚 Testing PubMedQA ({num_samples} samples)...")
        
        try:
            dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
            test_samples = dataset.shuffle(seed=42).select(range(num_samples))
        except Exception as e:
            print(f"❌ Could not load PubMedQA: {e}")
            return None
        
        correct = 0
        total = 0
        results = []
        
        for sample in tqdm(test_samples, desc="PubMedQA"):
            try:
                contexts = sample["context"]["contexts"]
                # Take only first context and limit length
                context_text = contexts[0][:300] if contexts else ""
                question = sample["question"]
                correct_answer = sample["final_decision"].lower()
                
                # Very simple prompt
                prompt = f"Context: {context_text}\nQuestion: {question}\nAnswer (yes/no/maybe):"
                
                response = self.generate_response(prompt, max_new_tokens=10)
                response_lower = response.lower().strip()
                
                # Extract answer
                if "yes" in response_lower:
                    predicted = "yes"
                elif "no" in response_lower:
                    predicted = "no"
                elif "maybe" in response_lower:
                    predicted = "maybe"
                else:
                    predicted = "unknown"
                
                is_correct = predicted == correct_answer
                if is_correct:
                    correct += 1
                total += 1
                
                results.append({
                    "question": question[:100] + "...",  # Truncate for storage
                    "correct_answer": correct_answer,
                    "predicted": predicted,
                    "correct": is_correct
                })
                
                # Print progress
                if total % 5 == 0:
                    print(f"Progress: {correct}/{total} correct so far")
                
            except Exception as e:
                print(f"Error processing sample {total}: {e}")
                continue
        
        accuracy = correct / total if total > 0 else 0
        print(f"📊 PubMedQA Accuracy: {accuracy:.3f} ({correct}/{total})")
        
        return {
            "benchmark": "PubMedQA",
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "results": results
        }
    
    def test_medical_reasoning(self):
        """Test on custom medical reasoning questions"""
        print("🧠 Testing Medical Reasoning...")
        
        reasoning_tasks = [
            {
                "question": "Patient has chest pain and shortness of breath. Most likely diagnosis?",
                "options": "A) Anxiety B) Heart attack C) Indigestion D) Cold",
                "correct": "B"
            },
            {
                "question": "Which medication should asthma patients avoid?",
                "options": "A) Albuterol B) Beta-blockers C) Steroids D) Antibiotics",
                "correct": "B"
            },
            {
                "question": "First-line treatment for Type 2 diabetes?",
                "options": "A) Insulin B) Metformin C) Diet only D) Surgery",
                "correct": "B"
            }
        ]
        
        correct = 0
        results = []
        
        for i, task in enumerate(reasoning_tasks):
            # Simple prompt
            prompt = f"Question: {task['question']}\n{task['options']}\nAnswer:"
            
            response = self.generate_response(prompt, max_new_tokens=5)
            predicted = self.extract_answer_choice(response)
            
            if predicted is None:
                predicted = "A"  # Default guess
            
            is_correct = predicted == task["correct"]
            if is_correct:
                correct += 1
            
            print(f"\nQ{i+1}: {task['question']}")
            print(f"Response: '{response}' -> Predicted: {predicted}, Correct: {task['correct']} {'✅' if is_correct else '❌'}")
            
            results.append({
                "question": task["question"],
                "correct_answer": task["correct"],
                "predicted": predicted,
                "response": response,
                "correct": is_correct
            })
        
        accuracy = correct / len(reasoning_tasks)
        print(f"\n📊 Medical Reasoning Accuracy: {accuracy:.3f} ({correct}/{len(reasoning_tasks)})")
        
        return {
            "benchmark": "Medical_Reasoning",
            "accuracy": accuracy,
            "correct": correct,
            "total": len(reasoning_tasks),
            "results": results
        }
    
    def run_quick_evaluation(self):
        """Run a quick evaluation on key benchmarks"""
        print("🚀 Starting Quick Medical Benchmark Evaluation")
        print("=" * 60)
        
        results = {}
        
        # Test Medical Reasoning first (faster)
        print("\n1. Testing Medical Reasoning...")
        reasoning_result = self.test_medical_reasoning()
        if reasoning_result:
            results["Medical_Reasoning"] = reasoning_result
        
        # Test PubMedQA with fewer samples
        print("\n2. Testing PubMedQA...")
        pubmed_result = self.test_pubmedqa(10)  # Reduced to 10 samples
        if pubmed_result:
            results["PubMedQA"] = pubmed_result
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 EVALUATION SUMMARY")
        print("=" * 60)
        
        summary = {}
        for benchmark_name, benchmark_data in results.items():
            accuracy = benchmark_data["accuracy"]
            summary[benchmark_name] = accuracy
            print(f"{benchmark_name:20}: {accuracy:.3f}")
        
        if summary:
            avg_accuracy = np.mean(list(summary.values()))
            print(f"\n{'Average Accuracy':20}: {avg_accuracy:.3f}")
            
            # Performance analysis
            print(f"\n📋 PERFORMANCE ANALYSIS:")
            print("-" * 40)
            
            if avg_accuracy >= 0.6:
                print("✅ Strong medical knowledge performance")
            elif avg_accuracy >= 0.4:
                print("⚠️ Moderate medical knowledge, room for improvement")
            else:
                print("❌ Needs significant improvement in medical knowledge")
        
        # Save results
        with open("quick_benchmark_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✅ Results saved to: quick_benchmark_results.json")
        
        return results, summary

def main():
    # Update this path to match your model
    model_path = "Cshavi/Gemma-2-2b-HealthCareMagic-v2"
    
    print("Starting medical benchmark evaluation...")
    print("This should take about 5-10 minutes.")
    
    try:
        # Initialize evaluator
        evaluator = MedicalBenchmarkEvaluator(model_path)
        
        # Run evaluation
        results, summary = evaluator.run_quick_evaluation()
        
        print("\n🎉 Benchmark evaluation completed successfully!")
        return results, summary
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        print("Check your model path and try again.")
        return None, None

if __name__ == "__main__":
    results, summary = main()