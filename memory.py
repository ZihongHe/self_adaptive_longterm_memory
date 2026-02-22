"""
Parametric memory management via LoRA adapters.

This module provides classes for dynamic loading, creation, training,
and deletion of LoRA adapters. It encapsulates the parametric memory
system that stores conversation knowledge in adapter weights.
"""

import random
import traceback
import torch
import torch.nn as nn
import transformers
from typing import List, Dict, Any, Optional
from peft import LoraConfig, get_peft_model
import numpy as np
import gc
import time
from torch.utils.data import DataLoader

from utils import ConversationDataset, collate_fn

class LoRAModelLoader:
    """Utility class for dynamic loading and management of LoRA models."""
    
    def __init__(self, base_model, lora_rank: int = 8, lora_alpha: int = 16, lora_dropout: float = 0.1):
        self.base_model = base_model
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
        
        self.lora_models = {}
        
    def get_lora_model(self, adapter_name: str, adapter_info: Optional[Dict] = None):
        if adapter_name in self.lora_models:
            return self.lora_models[adapter_name]
        
        lora_config = LoraConfig(
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            target_modules=self.target_modules,
            lora_dropout=self.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        lora_model = get_peft_model(self.base_model, lora_config)
        
        if adapter_info and adapter_info.get("state_dict"):
            try:
                lora_model.load_state_dict(adapter_info["state_dict"], strict=False)
                print(f"  ✅ Loaded LoRA weights for adapter: {adapter_name}")
            except Exception as e:
                print(f"  ⚠️ Failed to load LoRA weights for {adapter_name}: {e}")
        
        self.lora_models[adapter_name] = lora_model
        return lora_model
    
    def clear_cache(self):
        self.lora_models.clear()
        torch.cuda.empty_cache()


class EnhancedParametricMemory:
    """Enhanced parametric memory system with memory optimization."""
    
    def __init__(self, 
                 base_model_name: str = "model/Meta-Llama-3-8B-Instruct",
                 lora_rank: int = 8,
                 lora_alpha: int = 16,
                 lora_dropout: float = 0.1,
                 target_modules: List[str] = None,
                 max_adapters: int = 10,
                 training_learning_rate: float = 2e-4,
                 training_num_epochs: int = 1,
                 training_batch_size: int = 2,
                 max_training_samples: int = 30):
        
        self.base_model_name = base_model_name
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.max_adapters = max_adapters
        self.training_learning_rate = training_learning_rate
        self.training_num_epochs = training_num_epochs
        self.training_batch_size = training_batch_size
        self.max_training_samples = max_training_samples
        
        if target_modules is None:
            self.target_modules = [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]
        
        self.adapters: Dict[str, Dict[str, Any]] = {}
        self.active_adapter: Optional[str] = None
        self.active = False
        self.model: Optional[nn.Module] = None
        self.tokenizer: Optional[transformers.PreTrainedTokenizer] = None
        self.adapter_counter = 0
        self.current_iteration_adapters: List[str] = []
        self.usage_stats: Dict[str, Dict[str, Any]] = {}
        
        self.behavior_influence: Dict[str, float] = {
            "storage_influence": 0.0,
            "retrieval_influence": 0.0,
            "forgetting_influence": 0.0,
            "last_decision_effect": 0.0,
            "training_operations": 0,
            "total_training_tokens": 0,
        }
        
        self.path_adapters: Dict[str, List[str]] = {}
        self.branch_adapter_map: Dict[str, str] = {}
        
        self.forgotten_adapters: List[str] = []
        self.deleted_adapters_count = 0
        
    def set_model(self, model: nn.Module):
        """Set base model"""
        self.model = model
    
    def set_tokenizer(self, tokenizer: transformers.PreTrainedTokenizer):
        """Set tokenizer"""
        self.tokenizer = tokenizer
    
    def reset_for_new_session(self):
        """Completely reset parametric memory for new haystack_sessions."""
        print(f"  🔄 Resetting parametric memory: clearing all {len(self.adapters)} LoRA adapters")
        
        self.adapters.clear()
        self.active_adapter = None
        self.active = False
        self.adapter_counter = 0
        self.current_iteration_adapters.clear()
        self.usage_stats.clear()
        
        self.behavior_influence = {
            "storage_influence": 0.0,
            "retrieval_influence": 0.0,
            "forgetting_influence": 0.0,
            "last_decision_effect": 0.0,
            "training_operations": 0,
            "total_training_tokens": 0,
        }
        
        self.path_adapters.clear()
        self.branch_adapter_map.clear()
        
        self.forgotten_adapters.clear()
        self.deleted_adapters_count = 0
        
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
    def reset_iteration_adapters(self):
        """Reset adapters for current iteration"""
        self.current_iteration_adapters = []
        self.path_adapters = {}
        self.branch_adapter_map = {}
    
    def can_create_adapter(self) -> bool:
        """Check if new adapter can be created"""
        return len(self.current_iteration_adapters) < self.max_adapters
    
    def create_adapter(self, name: Optional[str] = None, path_id: str = "", branch_idx: int = 0) -> Optional[str]:
        """Create new LoRA adapter"""
        if not self.can_create_adapter():
            return None
            
        if self.model is None:
            raise ValueError("Model must be set before creating adapters")
        
        if name is None:
            name = f"adapter_{self.adapter_counter}"
            self.adapter_counter += 1
        
        lora_config = LoraConfig(
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            target_modules=self.target_modules,
            lora_dropout=self.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        self.adapters[name] = {
            "config": lora_config,
            "state_dict": None,
            "trained": False,
            "metadata": [],
            "training_data": [],
            "has_content": False,
            "initialized": False,
            "created_time": time.time(),
            "access_count": 0,
            "last_accessed": time.time(),
            "quality_score": 0.0,
            "relevance_score": 0.0,
            "path_id": path_id,
            "branch_idx": branch_idx,
            "deleted": False,
            "deleted_time": None,
            "deleted_by_path": None,
            "deleted_by_branch": None,
            "training_stats": {
                "total_training_samples": 0,
                "total_training_steps": 0,
                "last_training_time": None,
                "average_loss": 0.0,
                "perplexity": 0.0,
            }
        }
        
        self.current_iteration_adapters.append(name)
        
        if path_id:
            if path_id not in self.path_adapters:
                self.path_adapters[path_id] = []
            self.path_adapters[path_id].append(name)
        
        branch_key = f"{path_id}_{branch_idx}" if path_id else f"default_{branch_idx}"
        self.branch_adapter_map[branch_key] = name
        
        self.behavior_influence["storage_influence"] += 1.0
        
        return name
    
    def _format_conversation_for_training(self, conversation_data: Any) -> str:
        """Format conversation data for training with length limits."""
        if isinstance(conversation_data, dict):
            if "messages" in conversation_data:
                messages = conversation_data.get("messages", [])[:3]
                formatted_parts = []
                for msg in messages:
                    role = msg.get("role", "unknown")
                    content = msg.get("content", "")
                    if len(content) > 150:
                        content = content[:150] + "..."
                    formatted_parts.append(f"{role} says {content}")
                return "\n".join(formatted_parts)
            elif "text" in conversation_data:
                text = str(conversation_data["text"])
                return text[:500]
            else:
                return str(conversation_data)[:500]
        elif isinstance(conversation_data, str):
            return conversation_data[:500]
        elif isinstance(conversation_data, list):
            formatted_parts = []
            for item in conversation_data[:3]:
                if isinstance(item, dict) and "role" in item and "content" in item:
                    content = item.get("content", "")[:300]
                    formatted_parts.append(f"<{item['role']}> {content} </{item['role']}>")
                else:
                    formatted_parts.append(str(item)[:300])
            return "\n".join(formatted_parts[:3])
        else:
            return str(conversation_data)[:500]
    
    def _train_adapter_on_data(self, adapter_name: str, training_texts: List[str]) -> Dict[str, float]:
        """Train adapter on data with memory optimization."""
        if adapter_name not in self.adapters:
            return {"success": False, "reason": "adapter_not_found"}
        
        if self.model is None or self.tokenizer is None:
            return {"success": False, "reason": "model_or_tokenizer_not_set"}
        
        adapter = self.adapters[adapter_name]
        
        if adapter.get("deleted", False):
            return {"success": False, "reason": "adapter_deleted"}
        
        if not training_texts:
            return {"success": False, "reason": "no_training_data"}
        
        training_texts = training_texts[:min(self.max_training_samples, 20)]
        
        formatted_texts = []
        for text in training_texts:
            formatted = self._format_conversation_for_training(text)
            if formatted and len(formatted.strip()) > 10:
                formatted_texts.append(formatted)
        
        if not formatted_texts:
            return {"success": False, "reason": "no_valid_formatted_texts"}
        
        dataset = ConversationDataset(formatted_texts, self.tokenizer, max_length=256)
        
        if len(dataset) == 0:
            return {"success": False, "reason": "empty_dataset"}
        
        if len(dataset) > 15:
            indices = list(range(len(dataset)))
            random.shuffle(indices)
            indices = indices[:15]
            dataset = torch.utils.data.Subset(dataset, indices)
        
        dataloader = DataLoader(
            dataset,
            batch_size=min(self.training_batch_size, 2),
            shuffle=True,
            collate_fn=lambda batch: collate_fn(batch, self.tokenizer)
        )
        
        lora_loader = LoRAModelLoader(
            base_model=self.model,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout
        )
        
        try:
            adapter_info = {
                "config": adapter["config"],
                "state_dict": adapter["state_dict"]
            }
            
            lora_model = lora_loader.get_lora_model(adapter_name, adapter_info)
            lora_model.train()
            
            optimizer = torch.optim.AdamW(
                lora_model.parameters(),
                lr=self.training_learning_rate,
                weight_decay=0.01
            )
            
            total_loss = 0.0
            total_steps = 0
            device = next(lora_model.parameters()).device
            
            for epoch in range(min(self.training_num_epochs, 1)):
                epoch_loss = 0.0
                epoch_steps = 0
                
                for batch in dataloader:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)
                    
                    outputs = lora_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    
                    loss = outputs.loss
                    
                    optimizer.zero_grad()
                    loss.backward()
                    
                    torch.nn.utils.clip_grad_norm_(lora_model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    epoch_steps += 1
                    total_steps += 1
                    
                    del outputs, loss, input_ids, attention_mask, labels
                    if epoch_steps % 2 == 0:
                        torch.cuda.empty_cache()
                
                if epoch_steps > 0:
                    avg_epoch_loss = epoch_loss / epoch_steps
                    total_loss += avg_epoch_loss
            
            avg_loss = total_loss / max(self.training_num_epochs, 1)
            
            perplexity = np.exp(avg_loss) if avg_loss < 10 else float('inf')
            
            new_state_dict = {}
            for k, v in lora_model.state_dict().items():
                if 'lora' in k:
                    new_state_dict[k] = v.cpu()
            
            adapter["state_dict"] = new_state_dict
            adapter["trained"] = True
            adapter["has_content"] = True
            adapter["initialized"] = True
            
            adapter["training_data"] = []
            for i, text in enumerate(formatted_texts[:3]):
                adapter["training_data"].append({
                    "text": text[:100],
                    "timestamp": time.time()
                })
            
            current_quality = adapter.get("quality_score", 0.0)
            training_improvement = 0.1 * (1.0 - min(avg_loss / 5.0, 1.0))
            new_quality = max(0.0, min(1.0, current_quality + training_improvement))
            adapter["quality_score"] = new_quality
            
            adapter["training_stats"] = {
                "total_training_samples": adapter["training_stats"]["total_training_samples"] + len(formatted_texts),
                "total_training_steps": adapter["training_stats"]["total_training_steps"] + total_steps,
                "last_training_time": time.time(),
                "average_loss": avg_loss,
                "perplexity": perplexity,
            }
            
            self.behavior_influence["training_operations"] += 1
            self.behavior_influence["total_training_tokens"] += sum(len(t) for t in formatted_texts)
            
            del lora_model, optimizer, dataloader, dataset
            torch.cuda.empty_cache()
            gc.collect()
            
            return {
                "success": True,
                "average_loss": avg_loss,
                "perplexity": perplexity,
                "training_samples": len(formatted_texts),
                "training_steps": total_steps,
                "quality_improvement": training_improvement,
                "new_quality_score": new_quality,
            }
            
        except Exception as e:
            print(f"❌ Error training adapter {adapter_name}: {e}")
            traceback.print_exc()
            
            torch.cuda.empty_cache()
            gc.collect()
            
            return {"success": False, "reason": f"training_error: {str(e)}"}
    
    def get_adapter_for_branch(self, path_id: str, branch_idx: int) -> Optional[str]:
        """Get adapter for specific branch"""
        branch_key = f"{path_id}_{branch_idx}" if path_id else f"default_{branch_idx}"
        return self.branch_adapter_map.get(branch_key)
    
    def get_adapter_features(self, device: torch.device = None, path_id: str = "", branch_idx: int = -1) -> torch.Tensor:
        """Get stable adapter features"""
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        features = torch.zeros(self.max_adapters, device=device)
        
        if path_id and branch_idx >= 0:
            branch_adapter = self.get_adapter_for_branch(path_id, branch_idx)
            adapter_names = [branch_adapter] if branch_adapter else []
        elif path_id and path_id in self.path_adapters:
            adapter_names = self.path_adapters[path_id]
        else:
            adapter_names = self.current_iteration_adapters
        
        adapter_names = [name for name in adapter_names if name in self.adapters and not self.adapters[name].get("deleted", False)]
        
        for i in range(self.max_adapters):
            if i < len(adapter_names):
                adapter_name = adapter_names[i]
                if adapter_name in self.adapters:
                    adapter = self.adapters[adapter_name]
                    
                    if adapter.get("deleted", False):
                        features[i] = -1.0
                        continue
                    
                    if adapter.get("trained", False):
                        features[i] += 0.5
                    
                    training_samples = adapter["training_stats"].get("total_training_samples", 0)
                    features[i] += min(training_samples / 100.0, 0.5)
                    
                    features[i] += adapter.get("quality_score", 0.0) * 0.8
                    
                    features[i] += adapter.get("relevance_score", 0.0) * 0.3
                    
                    access_count = adapter.get("access_count", 0)
                    features[i] += min(access_count / 50.0, 1.0) * 0.2
                    
                    if path_id and adapter.get("path_id", "") == path_id:
                        features[i] += 0.2
                    
                    if branch_idx >= 0 and adapter.get("branch_idx", -1) == branch_idx:
                        features[i] += 0.1
                    
                    last_training = adapter["training_stats"].get("last_training_time")
                    if last_training:
                        time_diff = time.time() - last_training
                        freshness = max(0.0, 1.0 - min(time_diff / 3600.0, 1.0))
                        features[i] += freshness * 0.1
            else:
                features[i] = -0.5
        
        return features
    
    def get_adapter_available_mask(self, device: torch.device = None, path_id: str = "", branch_idx: int = -1) -> torch.Tensor:
        """Get adapter availability mask"""
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        mask = torch.zeros(self.max_adapters, device=device)
        
        if path_id and branch_idx >= 0:
            branch_adapter = self.get_adapter_for_branch(path_id, branch_idx)
            adapter_names = [branch_adapter] if branch_adapter else []
        elif path_id and path_id in self.path_adapters:
            adapter_names = self.path_adapters[path_id]
        else:
            adapter_names = self.current_iteration_adapters
        
        adapter_names = [name for name in adapter_names if name in self.adapters and not self.adapters[name].get("deleted", False)]
        
        for i, adapter_name in enumerate(adapter_names[:self.max_adapters]):
            if adapter_name in self.adapters:
                adapter = self.adapters[adapter_name]
                if adapter.get("trained", False) and adapter.get("has_content", False):
                    mask[i] = 1.0
        
        return mask
    
    def select_best_adapter_for_storage(self, text: str, path_id: str = "", branch_idx: int = -1) -> Optional[str]:
        """Select best adapter for storage"""
        adapter_list = []
        if path_id and path_id in self.path_adapters:
            adapter_list = self.path_adapters[path_id]
        else:
            adapter_list = self.current_iteration_adapters
        
        adapter_list = [name for name in adapter_list if name in self.adapters and not self.adapters[name].get("deleted", False)]
        
        if not adapter_list:
            return None
        
        best_adapter = None
        best_score = -float('inf')
        
        for adapter_name in adapter_list:
            if adapter_name in self.adapters:
                adapter = self.adapters[adapter_name]
                
                branch_bonus = 0.0
                if branch_idx >= 0 and adapter.get("branch_idx", -1) == branch_idx:
                    branch_bonus = 0.2
                
                quality = adapter.get("quality_score", 0.0)
                
                training_bonus = 0.3 if adapter.get("trained", False) else 0.0
                
                training_samples = adapter["training_stats"].get("total_training_samples", 0)
                capacity = max(0, self.max_training_samples - training_samples) / self.max_training_samples
                
                score = quality * 0.5 + training_bonus * 0.2 + capacity * 0.2 + branch_bonus
                
                if score > best_score:
                    best_score = score
                    best_adapter = adapter_name
        
        return best_adapter
    
    def compute_relevance(self, query: str, adapter_name: str) -> float:
        """Compute relevance between query and adapter content"""
        if adapter_name not in self.adapters:
            return 0.0
        
        adapter = self.adapters[adapter_name]
        
        if adapter.get("deleted", False):
            return 0.0
        
        training_data = adapter.get("training_data", [])
        
        if not training_data:
            return 0.0
        
        query_lower = query.lower()
        relevant_count = 0
        
        for entry in training_data[-10:]:
            entry_text = str(entry.get("text", "")).lower()
            if any(keyword in query_lower for keyword in ["what", "how", "why", "when", "where", "who", "which"]):
                if any(keyword in entry_text for keyword in ["what", "how", "why", "when", "where", "who", "which"]):
                    relevant_count += 1
            
            common_words = set(query_lower.split()) & set(entry_text.split())
            if len(common_words) > 2:
                relevant_count += 1
        
        relevance = min(relevant_count / 10.0, 1.0)
        
        self.adapters[adapter_name]["relevance_score"] = (
            0.9 * adapter.get("relevance_score", 0.0) + 0.1 * relevance
        )
        
        return relevance
    
    def update_adapter_quality(self, adapter_name: str, delta: float):
        """Update adapter quality score"""
        if adapter_name in self.adapters:
            adapter = self.adapters[adapter_name]
            if adapter.get("deleted", False):
                return
            current_quality = adapter.get("quality_score", 0.0)
            new_quality = max(0.0, min(1.0, current_quality + delta))
            adapter["quality_score"] = new_quality
    
    def write_to_adapter(self, conversation_data: Any, adapter_name: str) -> bool:
        """Write conversation to adapter and perform training."""
        if adapter_name not in self.adapters:
            return False
        
        adapter = self.adapters[adapter_name]
        
        if adapter.get("deleted", False):
            return False
        
        formatted_text = self._format_conversation_for_training(conversation_data)
        
        if not formatted_text or len(formatted_text.strip()) < 10:
            return False
        
        training_texts = []
        
        for entry in adapter.get("training_data", [])[-10:]:
            if "text" in entry:
                training_texts.append(entry["text"])
        
        training_texts.append(formatted_text)
        
        if len(training_texts) > self.max_training_samples:
            training_texts = training_texts[-self.max_training_samples:]
        
        training_result = self._train_adapter_on_data(adapter_name, training_texts)
        
        if training_result.get("success", False):
            adapter["access_count"] += 1
            adapter["last_accessed"] = time.time()
            
            if training_result.get("average_loss", 1.0) < 2.0:
                self.update_adapter_quality(adapter_name, 0.1)
            elif training_result.get("average_loss", 1.0) < 3.0:
                self.update_adapter_quality(adapter_name, 0.05)
            else:
                self.update_adapter_quality(adapter_name, 0.0)
            
            return True
        else:
            print(f"⚠️  Training failed for adapter {adapter_name}: {training_result.get('reason', 'unknown')}")
            return False
    
    def activate(self, adapter_name: str):
        """Activate adapter"""
        if adapter_name in self.adapters:
            adapter = self.adapters[adapter_name]
            
            if adapter.get("deleted", False):
                self.active = False
                self.active_adapter = None
                return
            
            self.active_adapter = adapter_name
            self.active = True
            
            self.adapters[adapter_name]["access_count"] += 1
            self.adapters[adapter_name]["last_accessed"] = time.time()
        else:
            self.active = False
            self.active_adapter = None
    
    def deactivate(self):
        """Deactivate adapter"""
        self.active = False
        self.active_adapter = None
    
    def get_memory_stats(self, path_id: str = "", branch_idx: int = -1) -> Dict[str, Any]:
        """Get memory statistics"""
        total_entries = 0
        adapters_with_content = 0
        total_access_count = 0
        avg_quality = 0.0
        deleted_count = 0
        total_training_samples = 0
        avg_perplexity = 0.0
        
        adapter_list = []
        if path_id and path_id in self.path_adapters:
            adapter_list = self.path_adapters[path_id]
        else:
            adapter_list = list(self.adapters.keys())
        
        for adapter_name in adapter_list:
            if adapter_name in self.adapters:
                adapter_info = self.adapters[adapter_name]
                if adapter_info.get("deleted", False):
                    deleted_count += 1
                    continue
                if adapter_info.get("has_content", False):
                    adapters_with_content += 1
                    total_entries += len(adapter_info.get("metadata", []))
                    total_access_count += adapter_info.get("access_count", 0)
                    avg_quality += adapter_info.get("quality_score", 0.0)
                    total_training_samples += adapter_info["training_stats"].get("total_training_samples", 0)
                    avg_perplexity += adapter_info["training_stats"].get("perplexity", 0.0)
        
        if adapters_with_content > 0:
            avg_quality /= adapters_with_content
            avg_perplexity /= adapters_with_content
        
        return {
            "total_adapters": len(adapter_list),
            "adapters_with_content": adapters_with_content,
            "total_entries": total_entries,
            "total_access_count": total_access_count,
            "average_quality": avg_quality,
            "current_iteration_adapters": len(self.current_iteration_adapters),
            "active_adapter": self.active_adapter,
            "is_active": self.active,
            "behavior_influence": self.behavior_influence,
            "path_count": len(self.path_adapters),
            "branch_adapter_map_size": len(self.branch_adapter_map),
            "deleted_adapters": deleted_count,
            "forgotten_adapters_count": len(self.forgotten_adapters),
            "training_stats": {
                "total_training_samples": total_training_samples,
                "total_training_operations": self.behavior_influence.get("training_operations", 0),
                "total_training_tokens": self.behavior_influence.get("total_training_tokens", 0),
                "average_perplexity": avg_perplexity,
            }
        }

