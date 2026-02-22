"""
Utility functions, configuration, and helper classes for the experiment.

This module centralises all auxiliary components including configuration
flags, data loading, prompt building, reward transformation, feature
extraction, and cumulative statistics tracking.
"""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
import os
import numpy as np
import gc
import time
import argparse
import transformers
from datetime import datetime
from torch.utils.data import Dataset
import psutil

IF_NO_PARAMETRIC = False
IF_NO_NON_PARAMETRIC = False

parser = argparse.ArgumentParser(description='Run multi-branch memory experiment')
parser.add_argument('--if_no_parametric', type=lambda x: (str(x).lower() == 'true'), 
                    default=IF_NO_PARAMETRIC, help='Disable parametric memory')
parser.add_argument('--if_no_non_parametric', type=lambda x: (str(x).lower() == 'true'), 
                    default=IF_NO_NON_PARAMETRIC, help='Disable non-parametric memory')
parser.add_argument('--num_repeat', type=int, default=4, help='Number of experiment repetitions')
parser.add_argument('--max_branches', type=int, default=20, help='Maximum number of branches')
parser.add_argument('--merge_similarity_threshold', type=float, default=0.9, help='Threshold for branch merging')
parser.add_argument('--num_samples', type=int, default=128, help='Number of samples to load from dataset')

args = parser.parse_args()

IF_NO_PARAMETRIC = args.if_no_parametric
IF_NO_NON_PARAMETRIC = args.if_no_non_parametric
NUM_SAMPLES = args.num_samples

print("\n" + "="*80)
print("Control experiment configuration:")
print(f"IF_NO_PARAMETRIC: {IF_NO_PARAMETRIC} - Disable parametric memory")
print(f"IF_NO_NON_PARAMETRIC: {IF_NO_NON_PARAMETRIC} - Disable non-parametric memory")
print(f"NUM_SAMPLES: {NUM_SAMPLES} - Number of samples to process")
print("="*80 + "\n")

experiment_config_str = (
    f"np{int(IF_NO_PARAMETRIC)}_"
    f"nnp{int(IF_NO_NON_PARAMETRIC)}_"
    f"samples{NUM_SAMPLES}"
)

def load_longmem_json(path: str) -> List[Dict[str, Any]]:
    """Load JSON data from file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def gumbel_sigmoid(logits: torch.Tensor, temperature: float = 1.0, hard: bool = False) -> torch.Tensor:
    """Gumbel-sigmoid relaxation for discrete binary decisions."""
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10) + 1e-10)
    y = logits + gumbel_noise
    y = torch.sigmoid(y / temperature)
    
    if hard:
        y_hard = (y > 0.5).float()
        y = y_hard - y.detach() + y
    
    return y

def gumbel_softmax(logits: torch.Tensor, temperature: float = 1.0, hard: bool = False) -> torch.Tensor:
    """Gumbel-softmax relaxation for categorical decisions."""
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10) + 1e-10)
    y = logits + gumbel_noise
    y = F.softmax(y / temperature, dim=-1)
    
    if hard:
        index = y.max(-1, keepdim=True)[1]
        y_hard = torch.zeros_like(y).scatter_(-1, index, 1.0)
        y = y_hard - y.detach() + y
    
    return y


class PromptBuilder:
    """Intelligent prompt builder ensuring questions are never truncated."""
    
    def __init__(self, tokenizer, max_total_tokens: int = 512, safety_margin: int = 20):
        self.tokenizer = tokenizer
        self.max_total_tokens = max_total_tokens
        self.safety_margin = safety_margin
        self.available_tokens = max_total_tokens - safety_margin
    
    def build_prompt_with_context(self, context_items: List[str], question: str) -> str:
        """Build prompt with context while ensuring question is fully preserved."""
        question_part = f"Q: {question}\nA:"
        
        question_tokens_with_specials = self.tokenizer.encode(
            question_part, 
            add_special_tokens=True,
            truncation=False
        )
        question_token_count = len(question_tokens_with_specials)
        
        remaining_tokens = self.available_tokens - question_token_count
        
        if remaining_tokens <= 0:
            print(f"⚠️ Warning: Question too long ({question_token_count} tokens), truncating...")
            truncated_question = self.tokenizer.decode(
                question_tokens_with_specials[:self.available_tokens - 10]
            ) + "..."
            return f"Q: {truncated_question}\nA:"
        
        context_parts = []
        current_tokens = 0
        
        for item in context_items:
            item_str = str(item)
            item_tokens = self.tokenizer.encode(
                f"- {item_str}\n",
                add_special_tokens=False
            )
            
            if current_tokens + len(item_tokens) <= remaining_tokens:
                context_parts.append(f"- {item_str}")
                current_tokens += len(item_tokens)
            else:
                if remaining_tokens - current_tokens > 20:
                    remaining = remaining_tokens - current_tokens
                    if remaining > 5:
                        truncated_item = self.tokenizer.decode(
                            item_tokens[:remaining - 5]
                        )
                        if truncated_item and len(truncated_item.strip()) > 0:
                            context_parts.append(f"- {truncated_item}...")
                break
        
        if context_parts:
            context = "\n".join(context_parts)
            prompt = f"Some previous conversations:\n{context}\n\n{question_part}"
        else:
            prompt = question_part
        
        final_tokens = self.tokenizer.encode(prompt, add_special_tokens=True)
        if len(final_tokens) > self.max_total_tokens:
            print(f"⚠️ Warning: Final prompt exceeds limit ({len(final_tokens)} > {self.max_total_tokens})")
            return self._truncate_prompt(prompt, question_part)
        
        return prompt
    
    def _truncate_prompt(self, full_prompt: str, question_part: str) -> str:
        """Truncate prompt while preserving question as much as possible."""
        question_tokens = self.tokenizer.encode(question_part, add_special_tokens=True)
        question_len = len(question_tokens)
        
        remaining = self.available_tokens - question_len - 10
        
        if remaining <= 0:
            return question_part
        
        if "Some previous conversations:" in full_prompt:
            parts = full_prompt.split("Some previous conversations:")
            if len(parts) > 1:
                context_part = parts[1].split(question_part)[0]
                context_tokens = self.tokenizer.encode(context_part, add_special_tokens=False)
                
                if len(context_tokens) > remaining:
                    truncated_context = self.tokenizer.decode(context_tokens[:remaining])
                    return f"Some previous conversations:\n{truncated_context}...\n\n{question_part}"
        
        return question_part
    
    def get_prompt_stats(self, prompt: str) -> Dict[str, Any]:
        """Get token statistics for a prompt."""
        tokens = self.tokenizer.encode(prompt, add_special_tokens=True)
        
        if "Q:" in prompt:
            parts = prompt.split("Q:")
            if len(parts) > 1:
                context_part = parts[0]
                question_part = "Q:" + parts[1].split("A:")[0] + "A:"
                
                context_tokens = self.tokenizer.encode(context_part, add_special_tokens=False)
                question_tokens = self.tokenizer.encode(question_part, add_special_tokens=False)
                
                return {
                    "total_tokens": len(tokens),
                    "context_tokens": len(context_tokens),
                    "question_tokens": len(question_tokens),
                    "question_encoded": question_part[:100] + "..." if len(question_part) > 100 else question_part,
                    "max_allowed": self.max_total_tokens,
                    "margin": self.safety_margin,
                    "is_question_complete": True
                }
        
        return {
            "total_tokens": len(tokens),
            "context_tokens": 0,
            "question_tokens": len(tokens),
            "max_allowed": self.max_total_tokens,
            "margin": self.safety_margin
        }


class CumulativeAverager:
    """Cumulative average calculator with EMA support."""
    
    def __init__(self, ema_alpha: float = 0.1):
        self.values = []
        self.cumulative_sums = []
        self.cumulative_averages = []
        self.ema_values = []
        self.current_sum = 0.0
        self.current_count = 0
        self.ema_alpha = ema_alpha
        self.current_ema = 0.0
        self.ema_initialized = False
    
    def add_value(self, value: float):
        """Add a value and update cumulative averages."""
        self.values.append(value)
        self.current_sum += value
        self.current_count += 1
        
        cumulative_sum = self.current_sum
        cumulative_avg = cumulative_sum / self.current_count
        
        self.cumulative_sums.append(cumulative_sum)
        self.cumulative_averages.append(cumulative_avg)
        
        if not self.ema_initialized:
            self.current_ema = value
            self.ema_initialized = True
        else:
            self.current_ema = self.ema_alpha * value + (1 - self.ema_alpha) * self.current_ema
        
        self.ema_values.append(self.current_ema)
        
        return cumulative_avg
    
    def add_values(self, values: List[float]):
        """Add multiple values."""
        results = []
        for value in values:
            results.append(self.add_value(value))
        return results
    
    def get_cumulative_averages(self) -> List[float]:
        return self.cumulative_averages
    
    def get_values(self) -> List[float]:
        return self.values
    
    def get_ema_values(self) -> List[float]:
        return self.ema_values
    
    def get_ema_alpha(self) -> float:
        return self.ema_alpha
    
    def reset(self):
        """Reset all accumulated data."""
        self.values = []
        self.cumulative_sums = []
        self.cumulative_averages = []
        self.ema_values = []
        self.current_sum = 0.0
        self.current_count = 0
        self.current_ema = 0.0
        self.ema_initialized = False
    
    def __len__(self):
        return len(self.values)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics of accumulated values."""
        if not self.values:
            return {}
        
        stats = {
            "count": len(self.values),
            "min_value": min(self.values),
            "max_value": max(self.values),
            "mean_value": np.mean(self.values),
            "std_value": np.std(self.values),
            "ema_alpha": self.ema_alpha,
            "final_ema": self.current_ema,
            "final_cumulative_avg": self.cumulative_averages[-1] if self.cumulative_averages else 0.0,
        }
        
        if self.ema_values:
            stats["ema_min"] = min(self.ema_values)
            stats["ema_max"] = max(self.ema_values)
            stats["ema_mean"] = np.mean(self.ema_values)
        
        return stats


class MultiRepeatCumulativeAverager:
    """Cumulative average calculator for multiple repeated experiments."""
    
    def __init__(self):
        self.repeat_data = []
        self.final_averages = None
        self.final_ema_averages = None
    
    def add_repeat(self, cumulative_averager: CumulativeAverager):
        """Add data from one repeat experiment."""
        self.repeat_data.append(cumulative_averager)
    
    def compute_final_averages(self) -> Tuple[List[float], List[float]]:
        """Compute final averages across all repeats."""
        if not self.repeat_data:
            print("  ⚠️  Warning: No repeat data available for cumulative averages")
            self.final_averages = []
            self.final_ema_averages = []
            return [], []
        
        valid_repeats = []
        for i, averager in enumerate(self.repeat_data):
            if len(averager.cumulative_averages) > 0:
                valid_repeats.append(averager)
            else:
                print(f"  ⚠️  Warning: Repeat {i} has empty cumulative averages")
        
        if not valid_repeats:
            print("  ⚠️  Warning: No valid repeat data with cumulative averages")
            self.final_averages = []
            self.final_ema_averages = []
            return [], []
        
        min_length = min(len(averager) for averager in valid_repeats)
        
        if min_length == 0:
            print("  ⚠️  Warning: Valid repeats have zero length cumulative averages")
            self.final_averages = []
            self.final_ema_averages = []
            return [], []
        
        final_averages = [0.0] * min_length
        final_ema_averages = [0.0] * min_length
        
        for i in range(min_length):
            position_values = []
            position_ema_values = []
            
            for averager in valid_repeats:
                if i < len(averager.cumulative_averages):
                    position_values.append(averager.cumulative_averages[i])
                    position_ema_values.append(averager.ema_values[i] if i < len(averager.ema_values) else averager.cumulative_averages[i])
            
            if position_values:
                final_averages[i] = np.mean(position_values)
                final_ema_averages[i] = np.mean(position_ema_values)
            else:
                final_averages[i] = 0.0
                final_ema_averages[i] = 0.0
        
        self.final_averages = final_averages
        self.final_ema_averages = final_ema_averages
        
        print(f"  ✅ Computed final cumulative averages: {len(final_averages)} values from {len(valid_repeats)} repeats")
        print(f"  ✅ Computed final EMA averages: {len(final_ema_averages)} values")
        
        return final_averages, final_ema_averages

    def get_repeat_stats(self) -> Dict[str, Any]:
        """Get statistics about repeats."""
        stats = {
            "num_repeats": len(self.repeat_data),
            "repeat_lengths": [len(averager) for averager in self.repeat_data],
            "final_averages_computed": self.final_averages is not None,
            "final_ema_averages_computed": self.final_ema_averages is not None,
        }
        
        if self.repeat_data:
            ema_alphas = [averager.ema_alpha for averager in self.repeat_data]
            stats["ema_alphas"] = ema_alphas
            stats["ema_alpha_consistent"] = all(alpha == ema_alphas[0] for alpha in ema_alphas)
            stats["ema_alpha"] = ema_alphas[0] if stats["ema_alpha_consistent"] else "mixed"
        
        if self.final_averages is not None:
            stats["final_averages_length"] = len(self.final_averages)
        
        if self.final_ema_averages is not None:
            stats["final_ema_averages_length"] = len(self.final_ema_averages)
        
        return stats
    
    def save_to_json(self, filepath: str):
        """Save data to JSON file."""
        data = {
            "num_repeats": len(self.repeat_data),
            "repeat_data": [
                {
                    "values": averager.values,
                    "cumulative_averages": averager.cumulative_averages,
                    "ema_values": averager.ema_values,
                    "ema_alpha": averager.ema_alpha
                }
                for averager in self.repeat_data
            ],
            "final_averages": self.final_averages if self.final_averages else [],
            "final_ema_averages": self.final_ema_averages if self.final_ema_averages else []
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def load_from_json(self, filepath: str):
        """Load data from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.repeat_data = []
        for repeat in data.get("repeat_data", []):
            ema_alpha = repeat.get("ema_alpha", 0.1)
            averager = CumulativeAverager(ema_alpha=ema_alpha)
            for value in repeat.get("values", []):
                averager.add_value(value)
            self.repeat_data.append(averager)
        
        self.final_averages = data.get("final_averages", [])
        self.final_ema_averages = data.get("final_ema_averages", [])


def compute_forgetting_quality(
    forgetting_decisions: List[Dict[str, Any]],
    generated_answers: List[str],
    target_answer: str,
    param_memory_stats: Dict[str, Any]
) -> List[float]:
    """Compute quality scores for forgetting decisions."""
    forgetting_quality_scores = []
    
    for i, decision in enumerate(forgetting_decisions):
        quality_score = 0.0
        
        if decision.get("forget", False):
            forget_reason = decision.get("reason", "")
            
            if "non_parametric" in forget_reason:
                non_param_idx = decision.get("forget_non_param_idx", -1)
                forgotten_content = decision.get("forgotten_non_param", "")
                
                if forgotten_content:
                    content_len = len(str(forgotten_content))
                    if content_len < 50:
                        quality_score += 0.3
                    if "error" in str(forgotten_content).lower():
                        quality_score += 0.2
                    
                    if i < len(generated_answers):
                        generated_answer = generated_answers[i]
                        if target_answer and generated_answer:
                            target_lower = str(target_answer).lower()
                            generated_lower = generated_answer.lower()
                            if target_lower in generated_lower or generated_lower in target_lower:
                                quality_score += 0.5
            
            elif "parametric" in forget_reason:
                adapter_name = decision.get("forget_param_name", "")
                
                if adapter_name:
                    adapter_quality = param_memory_stats.get("average_quality", 0.0)
                    if adapter_quality < 0.3:
                        quality_score += 0.4
                    
                    adapter_access_count = param_memory_stats.get("total_access_count", 0)
                    if adapter_access_count < 3:
                        quality_score += 0.3
                    
                    current_adapters = param_memory_stats.get("current_iteration_adapters", 0)
                    max_adapters = param_memory_stats.get("total_adapters", 10)
                    if current_adapters / max_adapters > 0.8:
                        quality_score += 0.3
            
            if i < len(generated_answers):
                generated_answer = generated_answers[i]
                if generated_answer and "ERROR" not in generated_answer:
                    quality_score += 0.2
        
        else:
            memory_usage = param_memory_stats.get("adapters_with_content", 0) / max(param_memory_stats.get("total_adapters", 10), 1)
            if memory_usage < 0.7:
                quality_score += 0.3
            
            if i < len(generated_answers):
                generated_answer = generated_answers[i]
                if generated_answer and "ERROR" not in generated_answer:
                    quality_score += 0.3
        
        forgetting_quality_scores.append(min(max(quality_score, 0.0), 1.0))
    
    return forgetting_quality_scores


def compute_target_logits_accuracy_with_model(
    model,
    tokenizer,
    prompt: str,
    target_answer: str,
    max_length: int = 512,
    use_length_normalization: bool = True
) -> Tuple[float, float, int]:
    """Compute logits difference and token accuracy for target answer."""
    prompt_str = str(prompt) if not isinstance(prompt, str) else prompt
    target_str = str(target_answer) if not isinstance(target_answer, str) else target_answer
    
    device = model.device
    model.eval()
    
    try:
        if "Q:" in prompt_str and "A:" in prompt_str:
            parts = prompt_str.split("Q:")
            if len(parts) > 1:
                prefix = parts[0]
                question_part = "Q:" + parts[1].split("A:")[0] + "A:"
                
                question_encoding = tokenizer(
                    question_part,
                    return_tensors="pt",
                    truncation=False,
                    add_special_tokens=True
                )
                
                question_tokens = question_encoding.input_ids
                question_len = question_tokens.shape[1]
                
                if prefix.strip():
                    remaining_tokens = max_length - question_len - 20
                    if remaining_tokens > 0:
                        prefix_encoding = tokenizer(
                            prefix,
                            return_tensors="pt",
                            truncation=True,
                            max_length=remaining_tokens,
                            add_special_tokens=False
                        )
                        
                        prompt_input_ids = torch.cat([
                            prefix_encoding.input_ids,
                            question_tokens
                        ], dim=1)
                        
                        prompt_attention_mask = torch.cat([
                            prefix_encoding.attention_mask,
                            torch.ones_like(question_tokens)
                        ], dim=1)
                        
                        prompt_encoding = {
                            "input_ids": prompt_input_ids.to(device),
                            "attention_mask": prompt_attention_mask.to(device)
                        }
                    else:
                        prompt_encoding = {
                            "input_ids": question_tokens.to(device),
                            "attention_mask": torch.ones_like(question_tokens).to(device)
                        }
                else:
                    prompt_encoding = {
                        "input_ids": question_tokens.to(device),
                        "attention_mask": torch.ones_like(question_tokens).to(device)
                    }
            else:
                prompt_encoding = tokenizer(
                    prompt_str,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_length // 2,
                    add_special_tokens=True
                ).to(device)
        else:
            prompt_encoding = tokenizer(
                prompt_str,
                return_tensors="pt",
                truncation=True,
                max_length=max_length // 2,
                add_special_tokens=True
            ).to(device)
        
        target_encoding = tokenizer(
            target_str,
            return_tensors="pt",
            truncation=True,
            max_length=max_length - prompt_encoding["input_ids"].shape[1],
            add_special_tokens=False
        ).to(device)
        
        target_len = target_encoding.attention_mask.sum().item()
        
        if target_len == 0:
            return float('inf'), 0.0, 0
        
        logits_differences = []
        token_accuracies = []
        
        with torch.no_grad():
            for i in range(target_len):
                partial_target_ids = target_encoding.input_ids[:, :i]
                input_ids = torch.cat([
                    prompt_encoding["input_ids"],
                    partial_target_ids
                ], dim=1)
                
                attention_mask = torch.cat([
                    prompt_encoding["attention_mask"],
                    target_encoding.attention_mask[:, :i]
                ], dim=1)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                next_token_logits = outputs.logits[:, -1, :]
                target_token_id = target_encoding.input_ids[0, i]
                
                probs = F.softmax(next_token_logits, dim=-1)
                target_prob = probs[0, target_token_id].item()
                
                nll = -np.log(target_prob + 1e-10)
                logits_differences.append(nll)
                
                predicted_token_id = next_token_logits.argmax(dim=-1).item()
                token_accuracy = 1.0 if predicted_token_id == target_token_id else 0.0
                token_accuracies.append(token_accuracy)
        
        if not logits_differences:
            return float('inf'), 0.0, 0
        
        if use_length_normalization and target_len > 0:
            if target_len > 1:
                weights = np.exp(np.linspace(0, 1, target_len))
                weights = weights / np.sum(weights)
                final_nll = np.average(logits_differences, weights=weights)
            else:
                final_nll = np.mean(logits_differences)
        else:
            final_nll = np.mean(logits_differences)
        
        avg_token_accuracy = np.mean(token_accuracies) if token_accuracies else 0.0
        
        return final_nll, avg_token_accuracy, target_len
        
    except Exception as e:
        print(f"Error computing target logits accuracy: {e}")
        return float('inf'), 0.0, 0
    

def evaluate_non_parametric_storage_decision(
    storage_decision: Dict[str, Any],
    generated_answer: str,
    target_answer: str,
    non_parametric_memory_size: int
) -> float:
    """Evaluate quality of non-parametric storage decisions."""
    quality_score = 0.5
    
    if storage_decision.get("store_to_non_parametric", False):
        store_prob = storage_decision.get("store_non_parametric_probability", 0.0)
        if store_prob > 0.7:
            quality_score += 0.2
        
        if generated_answer and target_answer:
            target_lower = str(target_answer).lower()
            generated_lower = generated_answer.lower()
            
            if target_lower in generated_lower or generated_lower in target_lower:
                quality_score += 0.3
        
        memory_usage = non_parametric_memory_size / 20.0
        if 0.3 <= memory_usage <= 0.8:
            quality_score += 0.2
        elif memory_usage > 0.9:
            quality_score -= 0.3
        
        reason = storage_decision.get("non_parametric_reason", "")
        if "high_quality" in reason or "important" in reason:
            quality_score += 0.1
        elif "full" in reason or "overflow" in reason:
            quality_score -= 0.2
    else:
        reason = storage_decision.get("non_parametric_reason", "")
        if "low_quality" in reason or "redundant" in reason:
            quality_score += 0.2
        elif "parametric_selected" in reason:
            quality_score += 0.1
        elif "both_memories_disabled" in reason:
            quality_score = 0.5
    
    return max(0.0, min(1.0, quality_score))


def extract_conversation_text(session: List[Dict[str, Any]], evaluate_quality: bool = False) -> Tuple[str, float]:
    """Extract text from conversation session and optionally evaluate quality."""
    conversation_text = ""
    
    if not isinstance(session, (list, tuple)):
        session = [session]
    
    quality_score = 0.5
    
    for message in session:
        if isinstance(message, dict):
            role = message.get("role", "")
            content = message.get("content", "")
            
            if evaluate_quality and content:
                content_str = str(content)
                
                length = len(content_str)
                if 50 <= length <= 500:
                    quality_score += 0.1
                elif length < 20:
                    quality_score -= 0.2
                
                info_density = 0
                for keyword in ["explain", "describe", "analyze", "compare", "because", "therefore", "however"]:
                    if keyword in content_str.lower():
                        info_density += 1
                
                quality_score += min(info_density * 0.05, 0.2)
                
                if any(marker in content_str for marker in ["1.", "2.", "3.", "- ", "* ", "• "]):
                    quality_score += 0.1
            
            conversation_text += f"{role}: {content}\n"
        else:
            content_str = str(message)
            conversation_text += content_str + "\n"
    
    text_str = str(conversation_text)
    if len(text_str) > 2000:
        text_str = text_str[:2000] + "..."
        quality_score -= 0.1
    
    quality_score = max(0.1, min(1.0, quality_score))
    
    return text_str.strip(), quality_score


def get_optimized_temperature(
    sample_idx: int, 
    initial_temp: float = 2.0,
    final_temp: float = 0.1,
    total_samples: int = 1000,
    decay_type: str = "cosine"
) -> float:
    """Get optimized temperature based on schedule."""
    progress = min(sample_idx / total_samples, 1.0)
    
    if decay_type == "cosine":
        return final_temp + 0.5 * (initial_temp - final_temp) * (1 + np.cos(np.pi * progress))
    
    elif decay_type == "exponential":
        decay_rate = (final_temp / initial_temp) ** (1.0 / total_samples)
        return initial_temp * (decay_rate ** sample_idx)
    
    elif decay_type == "linear":
        return initial_temp + (final_temp - initial_temp) * progress
    
    else:
        return final_temp + 0.5 * (initial_temp - final_temp) * (1 + np.cos(np.pi * progress))
    



class FeatureExtractor:
    """Extract features from text for adaptor decisions using pre-trained encoder."""
    
    def __init__(self, model_name: str = None, base_model: Optional[nn.Module] = None, use_encoder_only: bool = True):
        if base_model is not None:
            self.model = base_model
            self.model_config = base_model.config
            
            if use_encoder_only:
                if hasattr(base_model, 'transformer'):
                    self.encoder = base_model.transformer
                elif hasattr(base_model, 'model'):
                    self.encoder = base_model.model
                elif hasattr(base_model, 'encoder'):
                    self.encoder = base_model.encoder
                else:
                    self.encoder = base_model
            else:
                self.encoder = base_model
            
            self.hidden_dim = self.model_config.hidden_size
            self.has_final_norm = hasattr(self.encoder, 'norm') and self.encoder.norm is not None
                
        else:
            if model_name is None:
                raise ValueError("Either model_name or base_model must be provided")
                
            self.model = transformers.AutoModel.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True
            )
            self.encoder = self.model
            self.model_config = self.model.config
            self.hidden_dim = self.model_config.hidden_size
            self.has_final_norm = hasattr(self.encoder, 'norm') and self.encoder.norm is not None
        
        model_path = model_name if model_name else self.model_config._name_or_path
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        self.encoder.eval()
    
    @torch.no_grad()
    def extract_features(self, text: str, device: torch.device = None) -> torch.Tensor:
        if not isinstance(text, str):
            if isinstance(text, list):
                text = " ".join([str(item) for item in text])
            else:
                text = str(text)
        
        if len(str(text)) > 1000:
            text = str(text)[:1000]
        
        try:
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512,
                padding=True
            )
            
            inputs = {k: v.to(self.encoder.device) for k, v in inputs.items()}
            outputs = self.encoder(**inputs, output_hidden_states=True)
            
            if hasattr(outputs, 'last_hidden_state'):
                last_hidden = outputs.last_hidden_state
            elif hasattr(outputs, 'hidden_states'):
                hidden_states = outputs.hidden_states
                if hidden_states is not None and len(hidden_states) > 0:
                    last_hidden = hidden_states[-1]
                else:
                    raise ValueError("hidden_states is empty")
            else:
                raise AttributeError("No hidden states found in encoder output")
            
            if self.has_final_norm and hasattr(self.encoder, 'norm'):
                last_hidden = self.encoder.norm(last_hidden)
            
            if len(last_hidden.shape) == 3:
                features = last_hidden.mean(dim=1).squeeze()
            elif len(last_hidden.shape) == 2:
                features = last_hidden.mean(dim=0)
            else:
                features = last_hidden.flatten()
            
            features = features.flatten()
            
            if features.shape[0] != self.hidden_dim:
                if features.shape[0] > self.hidden_dim:
                    features = features[:self.hidden_dim]
                else:
                    padding = torch.zeros(self.hidden_dim - features.shape[0], 
                                        device=features.device, 
                                        dtype=features.dtype)
                    features = torch.cat([features, padding])
            
            if device is not None and features.device != device:
                features = features.to(device)
            
            return features.float()
            
        except Exception:
            if device is None:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            return torch.randn(self.hidden_dim, device=device, dtype=torch.float32)
    
    @torch.no_grad()
    def extract_features_batch(self, texts: List[str], device: torch.device = None) -> List[torch.Tensor]:
        if not texts:
            return []
        
        try:
            inputs = self.tokenizer(
                texts, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512,
                padding=True
            )
            
            inputs = {k: v.to(self.encoder.device) for k, v in inputs.items()}
            outputs = self.encoder(**inputs, output_hidden_states=True)
            
            if hasattr(outputs, 'last_hidden_state'):
                last_hidden = outputs.last_hidden_state
            elif hasattr(outputs, 'hidden_states'):
                hidden_states = outputs.hidden_states
                last_hidden = hidden_states[-1] if hidden_states is not None else None
            else:
                raise AttributeError("No hidden states found in encoder output")
            
            if last_hidden is None:
                raise ValueError("Failed to extract hidden states")
            
            if self.has_final_norm and hasattr(self.encoder, 'norm'):
                last_hidden = self.encoder.norm(last_hidden)
            
            features_batch = last_hidden.mean(dim=1)
            
            features_list = []
            for i in range(features_batch.shape[0]):
                feat = features_batch[i]
                if device is not None and feat.device != device:
                    feat = feat.to(device)
                features_list.append(feat.float())
            
            return features_list
            
        except Exception:
            if device is None:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            return [torch.randn(self.hidden_dim, device=device, dtype=torch.float32) for _ in texts]
        

class EnhancedRewardTransformer:
    """Enhanced reward transformer with length normalization and baseline adjustment."""
    
    def __init__(self, baseline_type: str = "ema", ema_alpha: float = 0.9):
        self.baseline_type = baseline_type
        self.ema_alpha = ema_alpha
        self.reward_history = []
        self.improvement_history = []
        self.baseline = 0.0
        self.baseline_history = []
        self.branch_rewards_history = []
        self.path_rewards_history = []
        
        self.length_adjustment_enabled = True
        self.length_adjustment_factor = 0.1
        self.token_info_history = []
        self.length_adjustment_history = []
        
    def update_baseline(self, reward: float):
        if self.baseline_type == "ema":
            self.baseline = self.ema_alpha * self.baseline + (1 - self.ema_alpha) * reward
        elif self.baseline_type == "mean":
            self.reward_history.append(reward)
            if len(self.reward_history) > 100:
                self.reward_history.pop(0)
            self.baseline = np.mean(self.reward_history)
        self.baseline_history.append(self.baseline)
        
    def record_branch_rewards(self, branch_rewards: List[float]):
        self.branch_rewards_history.append(branch_rewards)
        if len(self.branch_rewards_history) > 100:
            self.branch_rewards_history.pop(0)
    
    def record_path_rewards(self, path_rewards: List[float]):
        self.path_rewards_history.append(path_rewards)
        if len(self.path_rewards_history) > 100:
            self.path_rewards_history.pop(0)
    
    def compute_relative_rewards(self, branch_rewards: List[float]) -> List[float]:
        if not branch_rewards:
            return []
        
        avg_reward = np.mean(branch_rewards)
        std_reward = np.std(branch_rewards) if len(branch_rewards) > 1 else 1.0
        
        relative_rewards = [(r - avg_reward) / (std_reward + 1e-8) for r in branch_rewards]
        
        self.record_branch_rewards(branch_rewards)
        
        return relative_rewards
    
    def compute_path_normalized_rewards(self, path_rewards: List[float]) -> List[float]:
        if not path_rewards:
            return []
        
        min_val = min(path_rewards)
        if min_val < 0:
            shifted_rewards = [r - min_val + 1e-8 for r in path_rewards]
        else:
            shifted_rewards = [r + 1e-8 for r in path_rewards]
        
        inverted_rewards = [1.0 / r for r in shifted_rewards]
        
        max_val = max(inverted_rewards)
        if max_val > 0:
            normalized_rewards = [r / max_val for r in inverted_rewards]
        else:
            normalized_rewards = [0.0] * len(inverted_rewards)
        
        self.record_path_rewards(normalized_rewards)
        
        return normalized_rewards
    
    def _adjust_for_length(self, improvement: float, token_count_ratio: float) -> float:
        if not self.length_adjustment_enabled or token_count_ratio is None:
            return improvement
        
        adjustment_info = {
            "original_improvement": improvement,
            "token_count_ratio": token_count_ratio,
            "adjustment_factor": 0.0,
        }
        
        if token_count_ratio > 1.5:
            adjustment = self.length_adjustment_factor * min(token_count_ratio - 1.5, 1.0)
            adjusted_improvement = improvement * (1.0 + adjustment)
            adjustment_info["adjustment_factor"] = adjustment
            adjustment_info["adjustment_type"] = "long_answer_bonus"
            
        elif token_count_ratio < 0.7:
            adjustment = self.length_adjustment_factor * min(0.7 - token_count_ratio, 0.5)
            adjusted_improvement = improvement * (1.0 - adjustment)
            adjustment_info["adjustment_factor"] = -adjustment
            adjustment_info["adjustment_type"] = "short_answer_penalty"
            
        else:
            adjusted_improvement = improvement
            adjustment_info["adjustment_type"] = "no_adjustment"
        
        adjustment_info["adjusted_improvement"] = adjusted_improvement
        self.length_adjustment_history.append(adjustment_info)
        
        if len(self.length_adjustment_history) > 1000:
            self.length_adjustment_history = self.length_adjustment_history[-1000:]
        
        return adjusted_improvement
    
    def record_token_info(self, baseline_token_count: int, memory_token_count: int):
        token_info = {
            "baseline_token_count": baseline_token_count,
            "memory_token_count": memory_token_count,
            "token_count_ratio": memory_token_count / max(baseline_token_count, 1),
            "timestamp": time.time() if 'time' in globals() else 0,
        }
        self.token_info_history.append(token_info)
        
        if len(self.token_info_history) > 1000:
            self.token_info_history = self.token_info_history[-1000:]
    
    def transform(self, improvement: float, reward_type: str = "linear", **kwargs) -> Tuple[float, float]:
        token_count_ratio = kwargs.get('token_count_ratio', None)
        
        if token_count_ratio is not None and self.length_adjustment_enabled:
            baseline_token_count = kwargs.get('baseline_token_count', 0)
            memory_token_count = kwargs.get('memory_token_count', 0)
            if baseline_token_count > 0 and memory_token_count > 0:
                self.record_token_info(baseline_token_count, memory_token_count)
            
            adjusted_improvement = self._adjust_for_length(improvement, token_count_ratio)
        else:
            adjusted_improvement = improvement
        
        self.improvement_history.append(improvement)
        
        if reward_type == "linear":
            reward = adjusted_improvement
        elif reward_type == "clipped_linear":
            clip_min = kwargs.get('clip_min', -5.0)
            clip_max = kwargs.get('clip_max', 5.0)
            reward = np.clip(adjusted_improvement, clip_min, clip_max)
        elif reward_type == "signed_square":
            sign = 1.0 if adjusted_improvement >= 0 else -1.0
            reward = sign * np.sqrt(abs(adjusted_improvement))
        elif reward_type == "length_aware":
            if token_count_ratio is not None:
                length_factor = 1.0 / (1.0 + abs(token_count_ratio - 1.0))
                reward = adjusted_improvement * length_factor
            else:
                reward = adjusted_improvement
        else:
            reward = adjusted_improvement
        
        self.update_baseline(reward)
        normalized_reward = reward - self.baseline
        
        if len(self.reward_history) > 1000:
            self.reward_history = self.reward_history[-1000:]
            self.improvement_history = self.improvement_history[-1000:]
            self.baseline_history = self.baseline_history[-1000:]
        
        return reward, normalized_reward
    
    def enable_length_adjustment(self, enabled: bool = True):
        self.length_adjustment_enabled = enabled
        print(f"Length adjustment {'enabled' if enabled else 'disabled'}")
    
    def set_length_adjustment_factor(self, factor: float):
        self.length_adjustment_factor = max(0.0, min(1.0, factor))
        print(f"Length adjustment factor set to {factor}")
    
    def get_token_statistics(self) -> Dict[str, Any]:
        if not self.token_info_history:
            return {}
        
        token_count_ratios = [info["token_count_ratio"] for info in self.token_info_history]
        baseline_counts = [info["baseline_token_count"] for info in self.token_info_history]
        memory_counts = [info["memory_token_count"] for info in self.token_info_history]
        
        return {
            "avg_token_count_ratio": np.mean(token_count_ratios),
            "std_token_count_ratio": np.std(token_count_ratios),
            "min_token_count_ratio": np.min(token_count_ratios),
            "max_token_count_ratio": np.max(token_count_ratios),
            "avg_baseline_token_count": np.mean(baseline_counts),
            "avg_memory_token_count": np.mean(memory_counts),
            "long_answer_ratio": sum(1 for r in token_count_ratios if r > 1.5) / len(token_count_ratios),
            "short_answer_ratio": sum(1 for r in token_count_ratios if r < 0.7) / len(token_count_ratios),
            "total_samples": len(self.token_info_history),
        }
    
    def get_length_adjustment_statistics(self) -> Dict[str, Any]:
        if not self.length_adjustment_history:
            return {}
        
        adjustments = [info.get("adjustment_factor", 0.0) for info in self.length_adjustment_history]
        original_improvements = [info.get("original_improvement", 0.0) for info in self.length_adjustment_history]
        adjusted_improvements = [info.get("adjusted_improvement", 0.0) for info in self.length_adjustment_history]
        
        return {
            "avg_adjustment": np.mean(adjustments),
            "std_adjustment": np.std(adjustments),
            "max_positive_adjustment": max([a for a in adjustments if a > 0], default=0.0),
            "max_negative_adjustment": min([a for a in adjustments if a < 0], default=0.0),
            "adjustment_count": len(adjustments),
            "original_improvement_mean": np.mean(original_improvements),
            "adjusted_improvement_mean": np.mean(adjusted_improvements),
            "improvement_delta_mean": np.mean([adj - orig for adj, orig in zip(adjusted_improvements, original_improvements)]),
        }
    
    def get_stats(self) -> Dict[str, Any]:
        if not self.reward_history:
            return {}
        
        branch_stats = {}
        if self.branch_rewards_history:
            all_branch_rewards = [r for rewards in self.branch_rewards_history for r in rewards]
            if all_branch_rewards:
                branch_stats = {
                    "branch_reward_mean": np.mean(all_branch_rewards),
                    "branch_reward_std": np.std(all_branch_rewards),
                    "branch_reward_min": np.min(all_branch_rewards),
                    "branch_reward_max": np.max(all_branch_rewards),
                    "branch_count": len(all_branch_rewards),
                }
        
        path_stats = {}
        if self.path_rewards_history:
            all_path_rewards = [r for rewards in self.path_rewards_history for r in rewards]
            if all_path_rewards:
                path_stats = {
                    "path_reward_mean": np.mean(all_path_rewards),
                    "path_reward_std": np.std(all_path_rewards),
                    "path_reward_min": np.min(all_path_rewards),
                    "path_reward_max": np.max(all_path_rewards),
                    "path_count": len(all_path_rewards),
                }
        
        stats = {
            "reward_mean": np.mean(self.reward_history),
            "reward_std": np.std(self.reward_history),
            "reward_min": np.min(self.reward_history),
            "reward_max": np.max(self.reward_history),
            "improvement_mean": np.mean(self.improvement_history),
            "improvement_std": np.std(self.improvement_history),
            "baseline_current": self.baseline,
            "baseline_mean": np.mean(self.baseline_history) if self.baseline_history else 0,
            "length_adjustment_enabled": self.length_adjustment_enabled,
            "length_adjustment_factor": self.length_adjustment_factor,
        }
        
        token_stats = self.get_token_statistics()
        if token_stats:
            stats["token_statistics"] = token_stats
        
        adjustment_stats = self.get_length_adjustment_statistics()
        if adjustment_stats:
            stats["length_adjustment_statistics"] = adjustment_stats
        
        stats.update(branch_stats)
        stats.update(path_stats)
        
        return stats
    

class ConversationDataset(Dataset):
    """Conversation dataset for LoRA fine-tuning with memory optimization."""
    
    def __init__(self, conversation_texts: List[str], tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        max_text_length = 800
        
        for text in conversation_texts:
            if isinstance(text, dict) and "messages" in text:
                messages = text.get("messages", [])[:5]
                formatted_text = self._format_conversation_messages(messages)
            else:
                formatted_text = str(text)
            
            if len(formatted_text) > max_text_length:
                formatted_text = formatted_text[:max_text_length]
            
            try:
                tokens = tokenizer(
                    formatted_text,
                    truncation=True,
                    padding=False,
                    max_length=min(max_length, 256),
                    return_tensors=None
                )
                
                self.examples.append({
                    "input_ids": tokens["input_ids"],
                    "attention_mask": tokens["attention_mask"],
                    "labels": tokens["input_ids"].copy()
                })
            except Exception as e:
                print(f"  ⚠️  Tokenization failed: {e}, skipping")
                continue
        
        if len(self.examples) > 30:
            self.examples = self.examples[:30]
    
    def _format_conversation_messages(self, messages: List[Dict[str, str]]) -> str:
        """Format messages into training corpus with length limits."""
        formatted_parts = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if len(content) > 200:
                content = content[:200] + "..."
            formatted_parts.append(f"{role} says {content}")
        
        return "\n".join(formatted_parts)
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]


def collate_fn(batch, tokenizer):
    """Custom collation function for batching."""
    input_ids = [item["input_ids"] for item in batch]
    attention_mask = [item["attention_mask"] for item in batch]
    labels = [item["labels"] for item in batch]
    
    padded = tokenizer.pad(
        {"input_ids": input_ids},
        padding=True,
        return_tensors="pt"
    )
    
    padded_input_ids = padded["input_ids"]
    padded_attention_mask = padded["attention_mask"]
    
    padded_labels = tokenizer.pad(
        {"input_ids": labels},
        padding=True,
        return_tensors="pt"
    )["input_ids"]
    
    padded_labels[padded_labels == tokenizer.pad_token_id] = -100
    
    return {
        "input_ids": padded_input_ids,
        "attention_mask": padded_attention_mask,
        "labels": padded_labels
    }


def log_memory_usage(stage: str, threshold_mb: int = 8000):
    """Log memory usage and trigger garbage collection if above threshold."""
    try:
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        
        print(f"[{stage}] Memory usage: {memory_mb:.2f} MB")
        
        if memory_mb > threshold_mb:
            print(f"⚠️  High memory usage detected ({memory_mb:.2f} MB). Forcing garbage collection...")
            gc.collect()
            torch.cuda.empty_cache()
            
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            print(f"  ↳ After GC: {memory_mb:.2f} MB")
        
        return memory_mb
    except:
        return 0


def save_repeat_raw_stats_to_file(repeat_stat: Dict[str, Any], results_dir: str, repeat_idx: int, incremental_update: bool = False) -> Dict[str, Any]:
    """Save raw data for repeated experiments to file with support for incremental updates."""
    
    raw_file = os.path.join(results_dir, "repeat_stats", f"repeat_{repeat_idx}_raw.json")
    os.makedirs(os.path.dirname(raw_file), exist_ok=True)
    
    if incremental_update and os.path.exists(raw_file):
        try:
            with open(raw_file, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
            
            existing_data["cumulative_improvement_sequence"] = repeat_stat.get("cumulative_improvement_sequence", existing_data.get("cumulative_improvement_sequence", []))
            existing_data["cumulative_reward_sequence"] = repeat_stat.get("cumulative_reward_sequence", existing_data.get("cumulative_reward_sequence", []))
            existing_data["samples_processed"] = repeat_stat.get("samples_processed", existing_data.get("samples_processed", 0))
            existing_data["last_update"] = datetime.now().isoformat()
            
            temp_file = raw_file + '.tmp'
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, indent=2, ensure_ascii=False)
            os.replace(temp_file, raw_file)
            
            simplified_stat = {
                "repeat_idx": repeat_idx,
                "summary_stats": repeat_stat.get("summary_stats", {}),
                "cumulative_improvement_sequence": existing_data["cumulative_improvement_sequence"],
                "cumulative_reward_sequence": existing_data["cumulative_reward_sequence"],
                "samples_processed": existing_data["samples_processed"],
            }
            return simplified_stat
            
        except Exception as e:
            print(f"  ⚠️  Failed to incrementally update {raw_file}: {e}")
    
    simplified_stat = {
        "repeat_idx": repeat_idx,
        "summary_stats": repeat_stat["summary_stats"],
        "cumulative_improvement_sequence": repeat_stat.get("cumulative_improvement_sequence", []),
        "cumulative_reward_sequence": repeat_stat.get("cumulative_reward_sequence", []),
        "training_stats": repeat_stat.get("training_stats", {}),
        "samples_processed": repeat_stat.get("samples_processed", 0),
    }
    
    temp_file = raw_file + '.tmp'
    with open(temp_file, 'w', encoding='utf-8') as f:
        json.dump(simplified_stat, f, indent=2, ensure_ascii=False)
    os.replace(temp_file, raw_file)
    
    print(f"💾 Saved repeat {repeat_idx} raw stats to {raw_file}")
    
    return simplified_stat

