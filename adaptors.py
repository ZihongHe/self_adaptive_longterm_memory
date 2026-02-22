"""
Neural network modules for hierarchical memory decisions.

This module implements the adaptor networks that produce storage,
retrieval, and forgetting decisions using a two‑level hierarchical
structure with Gumbel‑softmax sampling.
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Tuple
import numpy as np
import torch.nn as nn
import time
import hashlib

from tree import MultiBranchDecisionTreeNode, MultiBranchDecisionTree, create_default_retrieval_decision, smart_align_nodes_and_decisions, create_default_storage_decision, create_default_forgetting_decision
from utils import gumbel_sigmoid, gumbel_softmax, IF_NO_PARAMETRIC, IF_NO_NON_PARAMETRIC

class EnhancedHierarchicalMemoryAdaptor(torch.nn.Module):
    """Enhanced hierarchical memory adaptor following MAP expansion principle with branch limits and merging."""
    
    def __init__(self, 
                 input_dim: int = 1536, 
                 max_adapters: int = 10,
                 adaptor_type: str = "storage",
                 max_branches: int = 64,
                 merge_similarity_threshold: float = 0.95):
        super().__init__()
        self.input_dim = input_dim
        self.max_adapters = max_adapters
        self.adaptor_type = adaptor_type
        
        self.level1_net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        self.level2_net = nn.Sequential(
            nn.Linear(input_dim + max_adapters, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )
        
        self.adapter_selector = nn.Sequential(
            nn.Linear(input_dim + max_adapters, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, max_adapters)
        )
        
        self.non_param_memory_evaluator = nn.Sequential(
            nn.Linear(input_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        self.best_branch_log_probs: List[torch.Tensor] = []
        self.best_branch_normalized_rewards: List[float] = []
        self.all_branch_log_probs: List[List[torch.Tensor]] = []
        self.all_branch_relative_rewards: List[List[float]] = []
        self.decision_tree = MultiBranchDecisionTree(
            max_branches=max_branches,
            merge_similarity_threshold=merge_similarity_threshold
        )
        
    def forward_level1(self, features: torch.Tensor) -> torch.Tensor:
        """Level 1 decision forward pass"""
        if len(features.shape) == 1:
            features = features.unsqueeze(0)
        
        device = next(self.parameters()).device
        if features.device != device:
            features = features.to(device)
        
        return self.level1_net(features).squeeze(-1)
    
    def decide_level1_with_map(self, features_list: List[torch.Tensor], 
                             temperature: float = 1.0, 
                             hard: bool = False) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """MAP-based Level 1 decision: make decisions for each feature."""
        decisions = []
        log_probs = []
        probabilities = []
        
        device = next(self.parameters()).device
        
        for features in features_list:
            features = features.to(device)
            logits = self.forward_level1(features)
            
            probs = torch.sigmoid(logits / temperature)
            decision = gumbel_sigmoid(logits, temperature, hard)
            
            log_prob = torch.where(
                decision > 0.5,
                torch.log(probs + 1e-10),
                torch.log(1 - probs + 1e-10)
            )
            
            decisions.append(decision)
            log_probs.append(log_prob)
            probabilities.append(probs)
        
        return decisions, log_probs, probabilities
    
    def forward_storage_level2(self, features: torch.Tensor, adapter_features: torch.Tensor) -> torch.Tensor:
        """Level 2 storage decision forward pass"""
        if len(features.shape) == 1:
            features = features.unsqueeze(0)
        
        device = next(self.parameters()).device
        if features.device != device:
            features = features.to(device)
            
        if len(adapter_features.shape) == 1:
            adapter_features = adapter_features.unsqueeze(0)
            
        combined = torch.cat([features, adapter_features], dim=-1)
        return self.level2_net(combined)
    
    def forward_adapter_selection(self, features: torch.Tensor, adapter_features: torch.Tensor, available_mask: torch.Tensor) -> torch.Tensor:
        """Adapter selection forward pass"""
        if len(features.shape) == 1:
            features = features.unsqueeze(0)
        
        device = next(self.parameters()).device
        if features.device != device:
            features = features.to(device)
        
        if adapter_features.device != device:
            adapter_features = adapter_features.to(device)
        
        if available_mask.device != device:
            available_mask = available_mask.to(device)
            
        if len(adapter_features.shape) == 1:
            adapter_features = adapter_features.unsqueeze(0)
            
        combined = torch.cat([features, adapter_features], dim=-1)
        logits = self.adapter_selector(combined)
        
        logits = logits + (available_mask - 1) * 1e10
        
        return logits

    def _compute_decision_hash(self, decision_type: str, decision_result: Any, features: torch.Tensor) -> str:
        """Compute decision hash value"""
        decision_str = f"{decision_type}_{decision_result}"
        if features is not None:
            features_data = features.cpu().numpy() if features.is_cuda else features.numpy()
            features_str = "_".join([f"{x:.3f}" for x in features_data[:5]])
            decision_str += f"_{features_str}"
        return hashlib.md5(decision_str.encode()).hexdigest()
    
    def create_decision_nodes_with_map_and_merge(self, 
                                               features_list: List[torch.Tensor],
                                               decision_type: str,
                                               decision_results: List[Any],
                                               log_probs: List[torch.Tensor],
                                               temperature: float,
                                               current_nodes: List[MultiBranchDecisionTreeNode]) -> List[MultiBranchDecisionTreeNode]:
        """MAP-based decision node creation with merge support using stable_id."""
        new_nodes = []
        
        for i, (features, decision_result, log_prob, current_node) in enumerate(
            zip(features_list, decision_results, log_probs, current_nodes)
        ):
            node = MultiBranchDecisionTreeNode(
                node_id=f"{decision_type}_{time.time()}_{i}",
                features=features.detach().cpu(),
                decision_type=decision_type,
                decision_result=decision_result,
                log_prob=log_prob.detach().cpu(),
                temperature=temperature,
                path_id=current_node.path_id + f"_{decision_type}" if current_node.path_id else f"p{self.decision_tree.path_counter}_{decision_type}",
                branch_idx=current_node.branch_idx,
                session_idx=current_node.session_idx,
                parametric_memory_state=current_node.parametric_memory_state,
                non_parametric_memory_state=current_node.non_parametric_memory_state,
                decision_hash=self._compute_decision_hash(decision_type, decision_result, features),
                merge_count=current_node.merge_count
            )
            new_nodes.append(node)
        
        return new_nodes
    
    def execute_storage_decisions_with_map(self, param_memory, 
                                        features_list: List[torch.Tensor],
                                        conversation_texts: List[str],
                                        temperature: float,
                                        current_nodes: List[MultiBranchDecisionTreeNode]) -> Tuple[List[Dict[str, Any]], List[MultiBranchDecisionTreeNode]]:
        """MAP-based execution of storage decisions using stable_id for alignment."""
        storage_decisions = []
        new_decision_nodes = []
        
        if not current_nodes:
            return storage_decisions, new_decision_nodes
        
        min_len = min(len(features_list), len(conversation_texts), len(current_nodes))
        
        if min_len < len(current_nodes):
            print(f"⚠️  Warning: Storage adaptor input lists length mismatch. Using min length: {min_len}")
            current_nodes = current_nodes[:min_len]
            features_list = features_list[:min_len]
            conversation_texts = conversation_texts[:min_len]
        
        child_nodes_list = []
        device = next(self.parameters()).device
        
        for i in range(min_len):
            features = features_list[i]
            conversation_text = conversation_texts[i]
            current_node = current_nodes[i]
            
            if features is None or current_node is None:
                default_decision = create_default_storage_decision(current_nodes[i] if i < len(current_nodes) else current_nodes[0])
                storage_decisions.append(default_decision)
                child_nodes_list.append([])
                continue
            
            child_nodes = []
            features = features.to(device) if features.device != device else features
            
            level1_logits = self.forward_level1(features.unsqueeze(0))
            level1_probs = torch.sigmoid(level1_logits / temperature)
            level1_decision = gumbel_sigmoid(level1_logits, temperature, hard=True)
            
            level1_log_prob = torch.where(
                level1_decision > 0.5,
                torch.log(level1_probs + 1e-10),
                torch.log(1 - level1_probs + 1e-10)
            )
            
            level1_node = MultiBranchDecisionTreeNode(
                node_id=f"storage_level1_{time.time()}_{i}",
                features=features.detach().cpu(),
                decision_type="storage_level1",
                decision_result=level1_decision.item() > 0.5,
                log_prob=level1_log_prob.detach().cpu(),
                temperature=temperature,
                path_id=current_node.path_id + "_storage_level1" if current_node.path_id else f"p{self.decision_tree.path_counter}_storage_level1",
                branch_idx=current_node.branch_idx,
                session_idx=current_node.session_idx,
                parametric_memory_state=current_node.parametric_memory_state,
                non_parametric_memory_state=current_node.non_parametric_memory_state,
                decision_hash=self._compute_decision_hash("storage_level1", level1_decision.item() > 0.5, features),
                merge_count=current_node.merge_count
            )
            
            child_nodes.append(level1_node)
            
            decision = {
                "store": level1_decision.item() > 0.5,
                "use_existing": False,
                "create_new": False,
                "selected_adapter": None,
                "store_to_non_parametric": False,
                "non_parametric_reason": "none",
                "store_non_parametric_probability": 0.0,
                "store_non_parametric_log_prob": 0.0,
                "reason": "none",
                "log_prob": level1_log_prob.item(),
                "store_probability": level1_probs.item(),
                "path_id": current_node.path_id,
                "branch_idx": current_node.branch_idx,
                "session_idx": current_node.session_idx,
                "merge_count": current_node.merge_count,
                "storage_to_non_parametric": False,
                "non_parametric_memory_usage": 0.0,
                "training_performed": False,
                "training_loss": 0.0,
                "stable_id": current_node.stable_id,
            }
            
            if decision["store"]:
                decision["reason"] = "store_decision_true"
                
                if not IF_NO_NON_PARAMETRIC:
                    non_param_features = self._extract_non_parametric_features(
                        current_node, device, max_memories=20
                    )
                    
                    combined_features = torch.cat([
                        features.unsqueeze(0),
                        non_param_features.unsqueeze(0)
                    ], dim=-1)
                    
                    if combined_features.shape[1] > self.level2_net[0].in_features:
                        combined_features = combined_features[:, :self.level2_net[0].in_features]
                    elif combined_features.shape[1] < self.level2_net[0].in_features:
                        padding = torch.zeros(
                            1, self.level2_net[0].in_features - combined_features.shape[1],
                            device=device
                        )
                        combined_features = torch.cat([combined_features, padding], dim=-1)
                    
                    non_param_logits = self.level2_net(combined_features)
                    non_param_probs = F.softmax(non_param_logits / temperature, dim=-1)
                    non_param_decision_onehot = gumbel_softmax(non_param_logits, temperature, hard=True)
                    non_param_decision = non_param_decision_onehot.argmax(dim=-1)
                    
                    non_param_log_prob = torch.log(non_param_probs[0, non_param_decision] + 1e-10)
                    
                    store_to_non_parametric = non_param_decision.item() == 0
                    decision["store_to_non_parametric"] = store_to_non_parametric
                    decision["store_non_parametric_probability"] = non_param_probs[0, 0].item()
                    decision["store_non_parametric_log_prob"] = non_param_log_prob.item()
                    decision["log_prob"] += non_param_log_prob.item()
                    
                    non_param_node = MultiBranchDecisionTreeNode(
                        node_id=f"storage_non_param_decision_{time.time()}_{i}",
                        features=features.detach().cpu(),
                        decision_type="storage_non_parametric_decision",
                        decision_result="store_to_non_parametric" if store_to_non_parametric else "store_to_parametric",
                        log_prob=non_param_log_prob.detach().cpu(),
                        temperature=temperature,
                        path_id=level1_node.path_id + "_non_param_decision",
                        branch_idx=current_node.branch_idx,
                        session_idx=current_node.session_idx,
                        parametric_memory_state=current_node.parametric_memory_state,
                        non_parametric_memory_state=current_node.non_parametric_memory_state,
                        decision_hash=self._compute_decision_hash(
                            "storage_non_parametric_decision",
                            "store_to_non_parametric" if store_to_non_parametric else "store_to_parametric",
                            features
                        ),
                        merge_count=current_node.merge_count
                    )
                    
                    child_nodes.append(non_param_node)
                    
                    if store_to_non_parametric:
                        decision["non_parametric_reason"] = "non_parametric_selected"
                        decision["storage_to_non_parametric"] = True
                        non_parametric_memory = current_node.non_parametric_memory_state or []
                        if len(non_parametric_memory) < 20:
                            truncated_text = conversation_text[:200]
                            if current_node.non_parametric_memory_state is None:
                                current_node.non_parametric_memory_state = []
                            current_node.non_parametric_memory_state.append(truncated_text)
                            level1_node.non_parametric_memory_state = current_node.non_parametric_memory_state
                            non_param_node.non_parametric_memory_state = current_node.non_parametric_memory_state
                        else:
                            decision["non_parametric_reason"] = "non_parametric_full"
                            decision["store_to_non_parametric"] = False
                            store_to_non_parametric = False
                    
                    if not store_to_non_parametric:
                        decision["non_parametric_reason"] = "parametric_selected"
                        
                        if not IF_NO_PARAMETRIC:
                            param_memory_state = current_node.parametric_memory_state
                            if param_memory_state is None:
                                param_memory_state = param_memory
                            
                            adapter_features = param_memory_state.get_adapter_features(
                                device=device,
                                path_id=current_node.path_id,
                                branch_idx=current_node.branch_idx
                            )
                            
                            param_logits = self.forward_storage_level2(features.unsqueeze(0), adapter_features.unsqueeze(0))
                            param_probs = F.softmax(param_logits / temperature, dim=-1)
                            param_decision_onehot = gumbel_softmax(param_logits, temperature, hard=True)
                            param_decision = param_decision_onehot.argmax(dim=-1)
                            
                            param_log_prob = torch.log(param_probs[0, param_decision] + 1e-10)
                            
                            if param_decision.item() == 0:
                                decision["use_existing"] = True
                                best_adapter = param_memory_state.select_best_adapter_for_storage(
                                    conversation_text, current_node.path_id, current_node.branch_idx
                                )
                                decision["selected_adapter"] = best_adapter
                                decision["reason"] = "use_existing_adapter"
                            else:
                                decision["create_new"] = True
                                new_adapter = param_memory_state.create_adapter(
                                    path_id=current_node.path_id, branch_idx=current_node.branch_idx
                                )
                                decision["selected_adapter"] = new_adapter
                                decision["reason"] = "create_new_adapter"
                            
                            decision["log_prob"] += param_log_prob.item()
                            
                            param_node = MultiBranchDecisionTreeNode(
                                node_id=f"storage_param_decision_{time.time()}_{i}",
                                features=features.detach().cpu(),
                                decision_type="storage_parametric_decision",
                                decision_result="create_new" if decision["create_new"] else "use_existing",
                                log_prob=param_log_prob.detach().cpu(),
                                temperature=temperature,
                                path_id=non_param_node.path_id + "_param_decision",
                                branch_idx=current_node.branch_idx,
                                session_idx=current_node.session_idx,
                                parametric_memory_state=param_memory_state,
                                non_parametric_memory_state=current_node.non_parametric_memory_state,
                                decision_hash=self._compute_decision_hash(
                                    "storage_parametric_decision",
                                    "create_new" if decision["create_new"] else "use_existing",
                                    features
                                ),
                                merge_count=current_node.merge_count
                            )
                            
                            child_nodes.append(param_node)
                            
                            if decision["store"] and decision["selected_adapter"] and not IF_NO_PARAMETRIC:
                                training_result = param_memory_state.write_to_adapter(
                                    conversation_text,
                                    decision["selected_adapter"]
                                )
                                
                                if isinstance(training_result, dict):
                                    decision["training_performed"] = training_result.get("success", False)
                                    decision["training_loss"] = training_result.get("average_loss", 0.0)
                                    decision["training_perplexity"] = training_result.get("perplexity", 0.0)
                                    decision["training_samples"] = training_result.get("training_samples", 0)
                                else:
                                    decision["training_performed"] = training_result
                                    decision["training_loss"] = 0.0
                                
                                if not decision["training_performed"]:
                                    decision["store"] = False
                                    decision["reason"] = "write_failed"
                                else:
                                    level1_node.parametric_memory_state = param_memory_state
                                    non_param_node.parametric_memory_state = param_memory_state
                                    param_node.parametric_memory_state = param_memory_state
                        else:
                            decision["non_parametric_reason"] = "parametric_disabled_fallback"
                            decision["store_to_non_parametric"] = True
                            decision["storage_to_non_parametric"] = True
                            
                            if current_node.non_parametric_memory_state is not None:
                                truncated_text = conversation_text[:200]
                                current_node.non_parametric_memory_state.append(truncated_text)
                                level1_node.non_parametric_memory_state = current_node.non_parametric_memory_state
                                non_param_node.non_parametric_memory_state = current_node.non_parametric_memory_state
                else:
                    decision["store_to_non_parametric"] = False
                    decision["non_parametric_reason"] = "non_parametric_disabled"
                    
                    if not IF_NO_PARAMETRIC:
                        param_memory_state = current_node.parametric_memory_state
                        if param_memory_state is None:
                            param_memory_state = param_memory
                        
                        adapter_features = param_memory_state.get_adapter_features(
                            device=device,
                            path_id=current_node.path_id,
                            branch_idx=current_node.branch_idx
                        )
                        
                        param_logits = self.forward_storage_level2(features.unsqueeze(0), adapter_features.unsqueeze(0))
                        param_probs = F.softmax(param_logits / temperature, dim=-1)
                        param_decision_onehot = gumbel_softmax(param_logits, temperature, hard=True)
                        param_decision = param_decision_onehot.argmax(dim=-1)
                        
                        param_log_prob = torch.log(param_probs[0, param_decision] + 1e-10)
                        
                        if param_decision.item() == 0:
                            decision["use_existing"] = True
                            best_adapter = param_memory_state.select_best_adapter_for_storage(
                                conversation_text, current_node.path_id, current_node.branch_idx
                            )
                            decision["selected_adapter"] = best_adapter
                            decision["reason"] = "use_existing_adapter"
                        else:
                            decision["create_new"] = True
                            new_adapter = param_memory_state.create_adapter(
                                path_id=current_node.path_id, branch_idx=current_node.branch_idx
                            )
                            decision["selected_adapter"] = new_adapter
                            decision["reason"] = "create_new_adapter"
                        
                        decision["log_prob"] += param_log_prob.item()
                        
                        param_node = MultiBranchDecisionTreeNode(
                            node_id=f"storage_param_decision_{time.time()}_{i}",
                            features=features.detach().cpu(),
                            decision_type="storage_parametric_decision",
                            decision_result="create_new" if decision["create_new"] else "use_existing",
                            log_prob=param_log_prob.detach().cpu(),
                            temperature=temperature,
                            path_id=level1_node.path_id + "_param_decision",
                            branch_idx=current_node.branch_idx,
                            session_idx=current_node.session_idx,
                            parametric_memory_state=param_memory_state,
                            non_parametric_memory_state=current_node.non_parametric_memory_state,
                            decision_hash=self._compute_decision_hash(
                                "storage_parametric_decision",
                                "create_new" if decision["create_new"] else "use_existing",
                                features
                            ),
                            merge_count=current_node.merge_count
                        )
                        
                        child_nodes.append(param_node)
                        
                        if decision["store"] and decision["selected_adapter"] and not IF_NO_PARAMETRIC:
                            training_result = param_memory_state.write_to_adapter(
                                conversation_text,
                                decision["selected_adapter"]
                            )
                            
                            if isinstance(training_result, dict):
                                decision["training_performed"] = training_result.get("success", False)
                                decision["training_loss"] = training_result.get("average_loss", 0.0)
                                decision["training_perplexity"] = training_result.get("perplexity", 0.0)
                                decision["training_samples"] = training_result.get("training_samples", 0)
                            else:
                                decision["training_performed"] = training_result
                                decision["training_loss"] = 0.0
                            
                            if not decision["training_performed"]:
                                decision["store"] = False
                                decision["reason"] = "write_failed"
                            else:
                                level1_node.parametric_memory_state = param_memory_state
                                param_node.parametric_memory_state = param_memory_state
                    else:
                        decision["store"] = False
                        decision["reason"] = "both_memories_disabled"
            else:
                decision["reason"] = "store_decision_false"
            
            non_parametric_memory = current_node.non_parametric_memory_state or []
            decision["non_parametric_memory_usage"] = len(non_parametric_memory) / 20.0
            
            storage_decisions.append(decision)
            
            if not child_nodes:
                child_nodes = [level1_node]
            
            child_nodes_list.append(child_nodes)
        
        if len(child_nodes_list) != min_len:
            print(f"❌ Error: Storage adaptor child nodes count mismatch: "
                  f"{len(child_nodes_list)} lists, {min_len} nodes expected")
            while len(child_nodes_list) < min_len:
                child_nodes_list.append([])
            child_nodes_list = child_nodes_list[:min_len]
        
        new_current_nodes = self.decision_tree.expand_all_current_with_map(child_nodes_list)
        
        if not new_current_nodes:
            return storage_decisions, new_decision_nodes
        
        new_decision_nodes = new_current_nodes
        
        if len(storage_decisions) != len(new_current_nodes):
            print(f"⚠️  Warning: Storage decisions count mismatch after expansion: "
                  f"{len(storage_decisions)} decisions, {len(new_current_nodes)} nodes")
            
            _, aligned_storage_decisions = smart_align_nodes_and_decisions(
                new_current_nodes, storage_decisions, create_default_storage_decision, "storage"
            )
            storage_decisions = aligned_storage_decisions
        
        return storage_decisions, new_decision_nodes

    def _extract_non_parametric_features(self, node: MultiBranchDecisionTreeNode, device: torch.device, max_memories: int = 20) -> torch.Tensor:
        """Extract features from non-parametric memory based on actual content."""
        non_parametric_memory = node.non_parametric_memory_state or []
        
        features = torch.zeros(self.input_dim, device=device)
        
        if non_parametric_memory:
            total_memories = len(non_parametric_memory)
            
            features[0] = min(total_memories / max_memories, 1.0)
            
            avg_length = 0.0
            question_count = 0
            answer_count = 0
            
            for mem in non_parametric_memory[:10]:
                mem_str = str(mem)
                avg_length += len(mem_str)
                
                if any(keyword in mem_str.lower() for keyword in ["what", "how", "why", "when", "where", "?"]):
                    question_count += 1
                elif any(keyword in mem_str.lower() for keyword in ["answer:", "explain", "describe", "because", "therefore"]):
                    answer_count += 1
            
            if total_memories > 0:
                avg_length /= min(total_memories, 10)
                features[1] = min(avg_length / 200.0, 1.0)
            
            if total_memories > 0:
                features[2] = min(question_count / 5.0, 1.0)
                features[3] = min(answer_count / 5.0, 1.0)
            
            if total_memories > 0:
                recent_mem = str(non_parametric_memory[-1]) if non_parametric_memory else ""
                recent_length = len(recent_mem)
                features[4] = min(recent_length / 200.0, 1.0)
                
                if "error" in recent_mem.lower():
                    features[5] = 1.0
                if "important" in recent_mem.lower() or "key" in recent_mem.lower():
                    features[6] = 1.0
            
            unique_keywords = set()
            for mem in non_parametric_memory[:5]:
                mem_str = str(mem).lower()
                for keyword in ["what", "how", "why", "when", "where", "who", "which"]:
                    if keyword in mem_str:
                        unique_keywords.add(keyword)
            
            features[7] = min(len(unique_keywords) / 7.0, 1.0)
        
        return features
    
    def execute_retrieval_decisions_with_map(self, param_memory,
                                        features_list: List[torch.Tensor],
                                        query_texts: List[str],
                                        temperature: float,
                                        current_nodes: List[MultiBranchDecisionTreeNode]) -> Tuple[List[Dict[str, Any]], List[float], List[MultiBranchDecisionTreeNode]]:
        """MAP-based execution of retrieval decisions using stable_id for alignment."""
        retrieval_decisions = []
        nll_adjustments = []
        new_decision_nodes = []
        
        if not current_nodes:
            return retrieval_decisions, nll_adjustments, new_decision_nodes
        
        min_len = min(len(features_list), len(query_texts), len(current_nodes))
        
        if min_len < len(current_nodes):
            print(f"⚠️  Warning: Retrieval adaptor input lists length mismatch. Using min length: {min_len}")
            current_nodes = current_nodes[:min_len]
            features_list = features_list[:min_len]
            query_texts = query_texts[:min_len]
        
        child_nodes_list = []
        device = next(self.parameters()).device
        
        for i in range(min_len):
            features = features_list[i]
            query_text = query_texts[i]
            current_node = current_nodes[i]
            
            if features is None or current_node is None:
                default_decision = create_default_retrieval_decision(current_nodes[i] if i < len(current_nodes) else current_nodes[0])
                retrieval_decisions.append(default_decision)
                nll_adjustments.append(0.0)
                child_nodes_list.append([])
                continue
            
            child_nodes = []
            features = features.to(device) if features.device != device else features
            
            level1_logits = self.forward_level1(features.unsqueeze(0))
            level1_probs = torch.sigmoid(level1_logits / temperature)
            level1_decision = gumbel_sigmoid(level1_logits, temperature, hard=True)
            
            level1_log_prob = torch.where(
                level1_decision > 0.5,
                torch.log(level1_probs + 1e-10),
                torch.log(1 - level1_probs + 1e-10)
            )
            
            level1_node = MultiBranchDecisionTreeNode(
                node_id=f"retrieval_level1_{time.time()}_{i}",
                features=features.detach().cpu(),
                decision_type="retrieval_level1",
                decision_result=level1_decision.item() > 0.5,
                log_prob=level1_log_prob.detach().cpu(),
                temperature=temperature,
                path_id=current_node.path_id + "_retrieval_level1" if current_node.path_id else f"p{self.decision_tree.path_counter}_retrieval_level1",
                branch_idx=current_node.branch_idx,
                session_idx=current_node.session_idx,
                parametric_memory_state=current_node.parametric_memory_state,
                non_parametric_memory_state=current_node.non_parametric_memory_state,
                decision_hash=self._compute_decision_hash("retrieval_level1", level1_decision.item() > 0.5, features),
                merge_count=current_node.merge_count
            )
            
            child_nodes.append(level1_node)
            
            decision = {
                "use_parametric": level1_decision.item() > 0.5 and not IF_NO_PARAMETRIC,
                "selected_adapter": None,
                "adapter_index": -1,
                "reason": "none",
                "log_prob": level1_log_prob.item(),
                "use_probability": level1_probs.item(),
                "nll_influence": 0.0,
                "path_id": current_node.path_id,
                "branch_idx": current_node.branch_idx,
                "session_idx": current_node.session_idx,
                "non_parametric_context": [],
                "non_param_relevance_scores": [],
                "non_param_decision_log_probs": [],
                "merge_count": current_node.merge_count,
                "stable_id": current_node.stable_id,
            }
            
            if decision["use_parametric"] and not IF_NO_PARAMETRIC:
                decision["reason"] = "use_decision_true"
                
                param_memory_state = current_node.parametric_memory_state
                if param_memory_state is None:
                    param_memory_state = param_memory
                
                adapter_features = param_memory_state.get_adapter_features(
                    device=device,
                    path_id=current_node.path_id,
                    branch_idx=current_node.branch_idx
                )
                
                available_mask = param_memory_state.get_adapter_available_mask(
                    device=device,
                    path_id=current_node.path_id,
                    branch_idx=current_node.branch_idx
                )
                
                adapter_logits = self.forward_adapter_selection(features.unsqueeze(0), adapter_features.unsqueeze(0), available_mask.unsqueeze(0))
                adapter_probs = F.softmax(adapter_logits / temperature, dim=-1)
                adapter_decision_onehot = gumbel_softmax(adapter_logits, temperature, hard=True)
                adapter_decision = adapter_decision_onehot.argmax(dim=-1)
                
                adapter_log_prob = torch.log(adapter_probs[0, adapter_decision] + 1e-10)
                
                decision["log_prob"] += adapter_log_prob.item()
                decision["adapter_index"] = adapter_decision.item()
                
                adapter_names = []
                if current_node.path_id and current_node.path_id in param_memory_state.path_adapters:
                    adapter_names = param_memory_state.path_adapters[current_node.path_id]
                else:
                    adapter_names = param_memory_state.current_iteration_adapters
                
                if adapter_decision.item() < len(adapter_names):
                    selected_adapter = adapter_names[adapter_decision.item()]
                    decision["selected_adapter"] = selected_adapter
                    decision["reason"] = f"selected_adapter_{adapter_decision.item()}"
                    
                    adapter_info = param_memory_state.adapters.get(selected_adapter, {})
                    quality = adapter_info.get("quality_score", 0.0)
                    relevance = param_memory_state.compute_relevance(query_text, selected_adapter)
                    
                    training_stats = adapter_info.get("training_stats", {})
                    perplexity = training_stats.get("perplexity", 10.0)
                    
                    training_factor = max(0.5, min(1.5, 1.0 / (perplexity / 5.0 + 0.1)))
                    decision["nll_influence"] = - (quality * 0.3 + relevance * 0.7) * 2.0 * training_factor
                    
                    if selected_adapter in param_memory_state.adapters:
                        param_memory_state.adapters[selected_adapter]["access_count"] += 1
                        param_memory_state.adapters[selected_adapter]["last_accessed"] = time.time()
                else:
                    decision["use_parametric"] = False
                    decision["reason"] = "invalid_adapter_selection"
                
                adapter_node = MultiBranchDecisionTreeNode(
                    node_id=f"adapter_selection_{time.time()}_{i}",
                    features=features.detach().cpu(),
                    decision_type="adapter_selection",
                    decision_result=decision["selected_adapter"] or "none",
                    log_prob=adapter_log_prob.detach().cpu(),
                    temperature=temperature,
                    path_id=level1_node.path_id + "_adapter_selection",
                    branch_idx=current_node.branch_idx,
                    session_idx=current_node.session_idx,
                    parametric_memory_state=param_memory_state,
                    non_parametric_memory_state=current_node.non_parametric_memory_state,
                    decision_hash=self._compute_decision_hash("adapter_selection", decision["selected_adapter"] or "none", features),
                    merge_count=current_node.merge_count
                )
                
                child_nodes.append(adapter_node)
                
                level1_node.parametric_memory_state = param_memory_state
                adapter_node.parametric_memory_state = param_memory_state
                
            else:
                decision["reason"] = "use_decision_false" if not IF_NO_PARAMETRIC else "parametric_disabled"
                decision["nll_influence"] = 0.3
            
            if not IF_NO_NON_PARAMETRIC:
                non_parametric_memory = current_node.non_parametric_memory_state or []
                
                if non_parametric_memory:
                    memory_features_list = []
                    memory_texts = []
                    
                    for mem_text in non_parametric_memory[:10]:
                        try:
                            mem_str = str(mem_text)
                            
                            mem_features = torch.zeros(self.input_dim, device=device)
                            
                            text_len = len(mem_str)
                            mem_features[0] = min(text_len / 200.0, 1.0)
                            
                            mem_lower = mem_str.lower()
                            
                            question_keywords = ["what", "how", "why", "when", "where", "who", "which", "?"]
                            question_score = 0.0
                            for kw in question_keywords:
                                if kw in mem_lower:
                                    question_score += 0.1
                            mem_features[1] = min(question_score, 1.0)
                            
                            answer_keywords = ["answer", "explain", "describe", "because", "therefore", "thus", "so"]
                            answer_score = 0.0
                            for kw in answer_keywords:
                                if kw in mem_lower:
                                    answer_score += 0.1
                            mem_features[2] = min(answer_score, 1.0)
                            
                            lines = mem_str.split('\n')
                            mem_features[3] = min(len(lines) / 5.0, 1.0)
                            
                            punctuation_count = mem_str.count('.') + mem_str.count('!') + mem_str.count('?')
                            mem_features[4] = min(punctuation_count / 5.0, 1.0)
                            
                            memory_features_list.append(mem_features)
                            memory_texts.append(mem_str[:100])
                            
                        except Exception:
                            mem_features = torch.zeros(self.input_dim, device=device)
                            mem_features[0] = 0.5
                            memory_features_list.append(mem_features)
                            memory_texts.append(str(mem_text)[:300])
                    
                    if memory_features_list:
                        selected_memories, relevance_scores, decision_log_probs = self.select_relevant_non_parametric_memories(
                            features,
                            memory_texts,
                            memory_features_list,
                            max_memories=3,
                            relevance_threshold=0.3,
                            temperature=temperature,
                            hard=True
                        )
                        
                        decision["non_parametric_context"] = selected_memories
                        decision["non_param_relevance_scores"] = relevance_scores
                        decision["non_param_decision_log_probs"] = [log_prob.item() for log_prob in decision_log_probs]
                        
                        total_non_param_log_prob = sum(decision_log_probs) if decision_log_probs else torch.tensor(0.0)
                        decision["log_prob"] += total_non_param_log_prob.item()
                        
                        if i < len(new_decision_nodes) and new_decision_nodes[i] is not None:
                            if hasattr(new_decision_nodes[i], 'log_prob') and new_decision_nodes[i].log_prob is not None:
                                new_log_prob = new_decision_nodes[i].log_prob + total_non_param_log_prob.detach().cpu()
                                new_decision_nodes[i].log_prob = new_log_prob
                    else:
                        decision["non_parametric_context"] = []
                else:
                    decision["non_parametric_context"] = []
            
            child_nodes_list.append(child_nodes)
            retrieval_decisions.append(decision)
            nll_adjustments.append(decision["nll_influence"])
        
        if len(child_nodes_list) != min_len:
            print(f"❌ Error: Retrieval adaptor child nodes count mismatch: "
                  f"{len(child_nodes_list)} lists, {min_len} nodes expected")
            while len(child_nodes_list) < min_len:
                child_nodes_list.append([])
            child_nodes_list = child_nodes_list[:min_len]
        
        new_current_nodes = self.decision_tree.expand_all_current_with_map(child_nodes_list)
        
        if not new_current_nodes:
            return retrieval_decisions, nll_adjustments, new_decision_nodes
        
        new_decision_nodes = new_current_nodes
        
        if len(retrieval_decisions) != len(new_current_nodes):
            print(f"⚠️  Warning: Retrieval decisions count mismatch after expansion: "
                  f"{len(retrieval_decisions)} decisions, {len(new_current_nodes)} nodes")
            
            _, aligned_retrieval_decisions = smart_align_nodes_and_decisions(
                new_current_nodes, retrieval_decisions, create_default_retrieval_decision, "retrieval"
            )
            retrieval_decisions = aligned_retrieval_decisions
        
        if len(nll_adjustments) != len(new_current_nodes):
            print(f"⚠️  Warning: NLL adjustments mismatch after alignment: "
                  f"{len(nll_adjustments)} adjustments, {len(new_current_nodes)} nodes")
            if len(nll_adjustments) < len(new_current_nodes):
                for _ in range(len(nll_adjustments), len(new_current_nodes)):
                    nll_adjustments.append(0.0)
            elif len(nll_adjustments) > len(new_current_nodes):
                nll_adjustments = nll_adjustments[:len(new_current_nodes)]
        
        return retrieval_decisions, nll_adjustments, new_decision_nodes
    
    def compute_reinforce_loss(self, use_all_branches: bool = True) -> torch.Tensor:
        """Compute REINFORCE loss"""
        if use_all_branches:
            return self._compute_all_branches_loss()
        else:
            return self._compute_best_branch_loss()
    
    def _compute_best_branch_loss(self) -> torch.Tensor:
        """Compute loss for best branch"""
        if not self.best_branch_log_probs or not self.best_branch_normalized_rewards:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        
        min_len = min(len(self.best_branch_log_probs), len(self.best_branch_normalized_rewards))
        if min_len == 0:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        
        log_probs = self.best_branch_log_probs[:min_len]
        normalized_rewards = torch.tensor(
            self.best_branch_normalized_rewards[:min_len],
            device=next(self.parameters()).device
        )
        
        log_probs_stack = torch.stack(log_probs).to(normalized_rewards.device)
        loss = -(log_probs_stack * normalized_rewards).mean()
        
        return loss
    
    def _compute_all_branches_loss(self) -> torch.Tensor:
        """Compute loss for all branches"""
        if not self.all_branch_log_probs or not self.all_branch_relative_rewards:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        
        all_losses = []
        device = next(self.parameters()).device
        
        for log_probs_list, relative_rewards_list in zip(self.all_branch_log_probs, self.all_branch_relative_rewards):
            if not log_probs_list or not relative_rewards_list:
                continue
                
            if len(log_probs_list) != len(relative_rewards_list):
                min_len = min(len(log_probs_list), len(relative_rewards_list))
                log_probs_list = log_probs_list[:min_len]
                relative_rewards_list = relative_rewards_list[:min_len]
            
            valid_log_probs = []
            valid_relative_rewards = []
            
            for j in range(len(log_probs_list)):
                lp = log_probs_list[j]
                rr = relative_rewards_list[j]
                
                if (lp is not None and 
                    rr is not None and
                    hasattr(lp, 'numel') and 
                    lp.numel() > 0 and
                    not torch.isnan(lp).any() and
                    not torch.isinf(lp).any()):
                    
                    if lp.numel() == 1:
                        valid_log_probs.append(lp.view(-1))
                    else:
                        valid_log_probs.append(lp.mean().view(-1))
                    
                    valid_relative_rewards.append(rr)
            
            if valid_log_probs and valid_relative_rewards and len(valid_log_probs) > 0:
                try:
                    log_probs_tensors = []
                    for lp in valid_log_probs:
                        if lp.device != device:
                            lp = lp.to(device)
                        if len(lp.shape) == 0:
                            lp = lp.view(1)
                        elif len(lp.shape) > 1:
                            lp = lp.view(-1)
                        log_probs_tensors.append(lp)
                    
                    if all(lp.shape == torch.Size([1]) for lp in log_probs_tensors):
                        log_probs_stack = torch.stack(log_probs_tensors)
                    else:
                        scalar_log_probs = [lp.mean().view(1) for lp in log_probs_tensors]
                        log_probs_stack = torch.stack(scalar_log_probs)
                    
                    relative_rewards = torch.tensor(
                        valid_relative_rewards,
                        device=device,
                        dtype=torch.float32
                    )
                    
                    if log_probs_stack.shape[0] == relative_rewards.shape[0]:
                        if len(relative_rewards.shape) == 1:
                            relative_rewards = relative_rewards.view(-1, 1)
                        if len(log_probs_stack.shape) == 1:
                            log_probs_stack = log_probs_stack.view(-1, 1)
                        
                        branch_loss = -(log_probs_stack * relative_rewards).mean()
                        
                        if not torch.isnan(branch_loss) and not torch.isinf(branch_loss):
                            all_losses.append(branch_loss)
                        else:
                            print(f"⚠️  Warning: Invalid loss value (NaN/Inf) for branch")
                    else:
                        print(f"⚠️  Warning: Shape mismatch after stacking: log_probs {log_probs_stack.shape}, rewards {relative_rewards.shape}")
                except RuntimeError as e:
                    print(f"⚠️  Warning: Error stacking tensors: {e}")
                    continue
        
        if not all_losses:
            return torch.tensor(0.0, device=device)
        
        valid_losses = []
        for loss in all_losses:
            if loss is not None and not torch.isnan(loss) and not torch.isinf(loss):
                valid_losses.append(loss)
        
        if not valid_losses:
            return torch.tensor(0.0, device=device)
        
        return torch.stack(valid_losses).mean()

    def clear_decision_history(self):
        """Clear decision history"""
        self.best_branch_log_probs.clear()
        self.best_branch_normalized_rewards.clear()
        self.all_branch_log_probs.clear()
        self.all_branch_relative_rewards.clear()
        if self.decision_tree.root:
            self.decision_tree.current_nodes = [self.decision_tree.root]
        else:
            self.decision_tree = MultiBranchDecisionTree(
                max_branches=self.decision_tree.max_branches,
                merge_similarity_threshold=self.decision_tree.merge_similarity_threshold
            )
    
    def get_gradient_norm(self) -> float:
        """Get gradient norm"""
        total_norm = 0.0
        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5 if total_norm > 0 else 0.0

    def evaluate_non_parametric_memory(self, 
                                     query_features: torch.Tensor,
                                     memory_features: torch.Tensor,
                                     temperature: float = 0.5,
                                     hard: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate whether a single non-parametric memory should be retrieved."""
        device = next(self.parameters()).device
        
        query_features = query_features.to(device).view(1, -1)
        memory_features = memory_features.to(device).view(1, -1)
        
        combined_features = torch.cat([query_features, memory_features], dim=-1)
        
        logits = self.non_param_memory_evaluator(combined_features)
        
        probs = torch.sigmoid(logits / temperature)
        decision = gumbel_sigmoid(logits, temperature, hard)
        
        log_prob = torch.where(
            decision > 0.5,
            torch.log(probs + 1e-10),
            torch.log(1 - probs + 1e-10)
        )
        
        return decision.squeeze(), log_prob.squeeze()
    
    def evaluate_non_parametric_memories_batch(self,
                                             query_features: torch.Tensor,
                                             memory_features_list: List[torch.Tensor],
                                             temperature: float = 0.5,
                                             hard: bool = False) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[float]]:
        """Batch evaluation of multiple non-parametric memories."""
        if not memory_features_list:
            return [], [], []
        
        decisions = []
        log_probs = []
        relevance_scores = []
        
        for memory_features in memory_features_list:
            if memory_features is not None:
                decision, log_prob = self.evaluate_non_parametric_memory(
                    query_features, memory_features, temperature, hard
                )
                relevance_score = torch.sigmoid(log_prob).item()
                
                decisions.append(decision)
                log_probs.append(log_prob)
                relevance_scores.append(relevance_score)
            else:
                decisions.append(torch.tensor(0.0, device=query_features.device))
                log_probs.append(torch.tensor(0.0, device=query_features.device))
                relevance_scores.append(0.0)
        
        return decisions, log_probs, relevance_scores
    
    def select_relevant_non_parametric_memories(self,
                                              query_features: torch.Tensor,
                                              memory_texts: List[str],
                                              memory_features_list: List[torch.Tensor],
                                              max_memories: int = 5,
                                              relevance_threshold: float = 0.3,
                                              temperature: float = 0.5,
                                              hard: bool = False) -> Tuple[List[str], List[float], List[torch.Tensor]]:
        """Select relevant non-parametric memories for retrieval."""
        if not memory_texts or not memory_features_list:
            return [], [], []
        
        decisions, log_probs, relevance_scores = self.evaluate_non_parametric_memories_batch(
            query_features, memory_features_list, temperature, hard
        )
        
        selected_memories = []
        selected_relevance_scores = []
        selected_log_probs = []
        
        for i, (decision, text, relevance_score, log_prob) in enumerate(
            zip(decisions, memory_texts, relevance_scores, log_probs)
        ):
            if decision.item() > 0.5 and relevance_score >= relevance_threshold:
                selected_memories.append(text)
                selected_relevance_scores.append(relevance_score)
                selected_log_probs.append(log_prob)
        
        if len(selected_memories) > max_memories:
            sorted_indices = np.argsort(selected_relevance_scores)[::-1]
            sorted_indices = sorted_indices[:max_memories]
            
            selected_memories = [selected_memories[i] for i in sorted_indices]
            selected_relevance_scores = [selected_relevance_scores[i] for i in sorted_indices]
            selected_log_probs = [selected_log_probs[i] for i in sorted_indices]
        
        return selected_memories, selected_relevance_scores, selected_log_probs

    def record_non_parametric_memory_decisions(self, path_id: str, log_probs: List[torch.Tensor], normalized_reward: float):
        """Record non-parametric memory decisions"""
        if log_probs:
            self.best_branch_log_probs.extend(log_probs)
            self.best_branch_normalized_rewards.extend([normalized_reward] * len(log_probs))
            
    def compute_non_parametric_memory_loss(self, retrieval_decisions: List[Dict[str, Any]], normalized_reward: float) -> torch.Tensor:
        """Compute loss for non-parametric memory selection."""
        device = next(self.parameters()).device
        total_loss = torch.tensor(0.0, device=device)
        count = 0
        
        for decision in retrieval_decisions:
            non_param_log_probs = decision.get("non_param_decision_log_probs", [])
            if non_param_log_probs:
                log_probs_tensor = torch.tensor(non_param_log_probs, device=device)
                branch_loss = -(log_probs_tensor * normalized_reward).mean()
                total_loss += branch_loss
                count += 1
        
        if count > 0:
            return total_loss / count
        else:
            return total_loss
        
    def extract_memory_text_features(self, memory_text: str, device: torch.device = None) -> torch.Tensor:
        """Extract features from memory text based on content."""
        if device is None:
            device = next(self.parameters()).device
        
        features = torch.zeros(self.input_dim, device=device)
        
        if not memory_text or not isinstance(memory_text, str):
            return features
        
        text = str(memory_text)
        
        text_len = len(text)
        features[0] = min(text_len / 500.0, 1.0)
        
        text_lower = text.lower()
        
        question_keywords = ["what", "how", "why", "when", "where", "who", "which", "?"]
        question_score = 0.0
        for kw in question_keywords:
            if kw in text_lower:
                question_score += 0.1
        features[1] = min(question_score, 1.0)
        
        answer_keywords = ["answer", "explain", "describe", "because", "therefore", "thus", "so"]
        answer_score = 0.0
        for kw in answer_keywords:
            if kw in text_lower:
                answer_score += 0.1
        features[2] = min(answer_score, 1.0)
        
        importance_keywords = ["important", "key", "crucial", "essential", "significant"]
        importance_score = 0.0
        for kw in importance_keywords:
            if kw in text_lower:
                importance_score += 0.2
        features[3] = min(importance_score, 1.0)
        
        lines = text.split('\n')
        features[4] = min(len(lines) / 10.0, 1.0)
        
        punctuation_count = text.count('.') + text.count('!') + text.count('?')
        features[5] = min(punctuation_count / 10.0, 1.0)
        
        if text_len > 0:
            uppercase_ratio = sum(1 for c in text if c.isupper()) / text_len
            features[6] = min(uppercase_ratio * 10.0, 1.0)
        
        digit_count = sum(c.isdigit() for c in text)
        features[7] = min(digit_count / 10.0, 1.0)
        
        return features
    

class ForgettingAdaptor(torch.nn.Module):
    """Forgetting adaptor: decides which non-parametric and parametric memories to forget."""
    
    def __init__(self, 
                 input_dim: int = 1536, 
                 max_adapters: int = 10,
                 max_non_param_memory: int = 3,
                 max_branches: int = 64,
                 merge_similarity_threshold: float = 0.95):
        super().__init__()
        self.input_dim = input_dim
        self.max_adapters = max_adapters
        self.max_non_param_memory = max_non_param_memory
        
        self.level1_net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        self.level2_net = nn.Sequential(
            nn.Linear(input_dim + max_adapters + max_non_param_memory, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )
        
        self.non_param_selector = nn.Sequential(
            nn.Linear(input_dim + max_non_param_memory, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, max_non_param_memory)
        )
        
        self.param_selector = nn.Sequential(
            nn.Linear(input_dim + max_adapters, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, max_adapters)
        )
        
        self.best_branch_log_probs: List[torch.Tensor] = []
        self.best_branch_normalized_rewards: List[float] = []
        self.all_branch_log_probs: List[List[torch.Tensor]] = []
        self.all_branch_relative_rewards: List[List[float]] = []
        self.decision_tree = MultiBranchDecisionTree(
            max_branches=max_branches,
            merge_similarity_threshold=merge_similarity_threshold
        )
        
    def forward_level1(self, features: torch.Tensor) -> torch.Tensor:
        """Level 1 decision forward pass"""
        if len(features.shape) == 1:
            features = features.unsqueeze(0)
        
        device = next(self.parameters()).device
        if features.device != device:
            features = features.to(device)
        
        return self.level1_net(features).squeeze(-1)
    
    def decide_level1_with_map(self, features_list: List[torch.Tensor], 
                             temperature: float = 1.0, 
                             hard: bool = False) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """MAP-based Level 1 decision: make decisions for each feature."""
        decisions = []
        log_probs = []
        probabilities = []
        
        device = next(self.parameters()).device
        
        for features in features_list:
            features = features.to(device)
            logits = self.forward_level1(features)
            
            probs = torch.sigmoid(logits / temperature)
            decision = gumbel_sigmoid(logits, temperature, hard)
            
            log_prob = torch.where(
                decision > 0.5,
                torch.log(probs + 1e-10),
                torch.log(1 - probs + 1e-10)
            )
            
            decisions.append(decision)
            log_probs.append(log_prob)
            probabilities.append(probs)
        
        return decisions, log_probs, probabilities
    
    def forward_level2(self, features: torch.Tensor, adapter_features: torch.Tensor, non_param_features: torch.Tensor) -> torch.Tensor:
        """Level 2 decision forward pass"""
        if len(features.shape) == 1:
            features = features.unsqueeze(0)
        
        device = next(self.parameters()).device
        if features.device != device:
            features = features.to(device)
            
        if len(adapter_features.shape) == 1:
            adapter_features = adapter_features.unsqueeze(0)
            
        if len(non_param_features.shape) == 1:
            non_param_features = non_param_features.unsqueeze(0)
        
        if features.shape[0] > 1 and adapter_features.shape[0] == 1:
            adapter_features = adapter_features.repeat(features.shape[0], 1)
        if features.shape[0] > 1 and non_param_features.shape[0] == 1:
            non_param_features = non_param_features.repeat(features.shape[0], 1)
            
        combined = torch.cat([features, adapter_features, non_param_features], dim=-1)
        return self.level2_net(combined)
    
    def decide_level2_with_map(self, features_list: List[torch.Tensor],
                             adapter_features_list: List[torch.Tensor],
                             non_param_features_list: List[torch.Tensor],
                             temperature: float = 1.0,
                             hard: bool = False) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """MAP-based Level 2 forgetting type decision."""
        decisions = []
        log_probs = []
        probabilities = []
        
        device = next(self.parameters()).device
        
        for features, adapter_features, non_param_features in zip(features_list, adapter_features_list, non_param_features_list):
            features = features.to(device)
            adapter_features = adapter_features.to(device)
            non_param_features = non_param_features.to(device)
            
            logits = self.forward_level2(features, adapter_features, non_param_features)
            
            probs = F.softmax(logits / temperature, dim=-1)
            decision_onehot = gumbel_softmax(logits, temperature, hard)
            
            if hard:
                decision = decision_onehot.argmax(dim=-1)
            else:
                decision = (decision_onehot[:, 1] > 0.5).long()
            
            log_prob = torch.log(probs[range(len(probs)), decision] + 1e-10)
            
            decisions.append(decision)
            log_probs.append(log_prob)
            probabilities.append(probs)
        
        return decisions, log_probs, probabilities
    
    def forward_non_param_selection(self, features: torch.Tensor, non_param_features: torch.Tensor, non_param_mask: torch.Tensor) -> torch.Tensor:
        """Non-parametric memory selection forward pass"""
        if len(features.shape) == 1:
            features = features.unsqueeze(0)
        
        device = next(self.parameters()).device
        if features.device != device:
            features = features.to(device)
        
        if non_param_features.device != device:
            non_param_features = non_param_features.to(device)
        
        if non_param_mask.device != device:
            non_param_mask = non_param_mask.to(device)
            
        if len(non_param_features.shape) == 1:
            non_param_features = non_param_features.unsqueeze(0)
        
        if features.shape[0] > 1 and non_param_features.shape[0] == 1:
            non_param_features = non_param_features.repeat(features.shape[0], 1)
            
        combined = torch.cat([features, non_param_features], dim=-1)
        logits = self.non_param_selector(combined)
        
        logits = logits + (non_param_mask - 1) * 1e10
        
        return logits
    
    def decide_non_param_selection_with_map(self, features_list: List[torch.Tensor],
                                          non_param_features_list: List[torch.Tensor],
                                          non_param_masks: List[torch.Tensor],
                                          temperature: float = 1.0,
                                          hard: bool = False) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """MAP-based non-parametric memory selection decision."""
        decisions = []
        log_probs = []
        probabilities = []
        
        device = next(self.parameters()).device
        
        for features, non_param_features, non_param_mask in zip(features_list, non_param_features_list, non_param_masks):
            features = features.to(device)
            non_param_features = non_param_features.to(device)
            non_param_mask = non_param_mask.to(device)
            
            logits = self.forward_non_param_selection(features, non_param_features, non_param_mask)
            
            probs = F.softmax(logits / temperature, dim=-1)
            decision_onehot = gumbel_softmax(logits, temperature, hard)
            
            if hard:
                decision = decision_onehot.argmax(dim=-1)
            else:
                decision = torch.multinomial(probs, 1).squeeze(-1)
            
            log_prob = torch.log(probs[range(len(probs)), decision] + 1e-10)
            
            decisions.append(decision)
            log_probs.append(log_prob)
            probabilities.append(probs)
        
        return decisions, log_probs, probabilities
    
    def forward_param_selection(self, features: torch.Tensor, adapter_features: torch.Tensor, param_mask: torch.Tensor) -> torch.Tensor:
        """Parametric memory selection forward pass"""
        if len(features.shape) == 1:
            features = features.unsqueeze(0)
        
        device = next(self.parameters()).device
        if features.device != device:
            features = features.to(device)
        
        if adapter_features.device != device:
            adapter_features = adapter_features.to(device)
        
        if param_mask.device != device:
            param_mask = param_mask.to(device)
            
        if len(adapter_features.shape) == 1:
            adapter_features = adapter_features.unsqueeze(0)
        
        if features.shape[0] > 1 and adapter_features.shape[0] == 1:
            adapter_features = adapter_features.repeat(features.shape[0], 1)
            
        combined = torch.cat([features, adapter_features], dim=-1)
        logits = self.param_selector(combined)
        
        logits = logits + (param_mask - 1) * 1e10
        
        return logits
    
    def decide_param_selection_with_map(self, features_list: List[torch.Tensor],
                                      adapter_features_list: List[torch.Tensor],
                                      param_masks: List[torch.Tensor],
                                      temperature: float = 1.0,
                                      hard: bool = False) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """MAP-based parametric memory selection decision."""
        decisions = []
        log_probs = []
        probabilities = []
        
        device = next(self.parameters()).device
        
        for features, adapter_features, param_mask in zip(features_list, adapter_features_list, param_masks):
            features = features.to(device)
            adapter_features = adapter_features.to(device)
            param_mask = param_mask.to(device)
            
            logits = self.forward_param_selection(features, adapter_features, param_mask)
            
            probs = F.softmax(logits / temperature, dim=-1)
            decision_onehot = gumbel_softmax(logits, temperature, hard)
            
            if hard:
                decision = decision_onehot.argmax(dim=-1)
            else:
                decision = torch.multinomial(probs, 1).squeeze(-1)
            
            log_prob = torch.log(probs[range(len(probs)), decision] + 1e-10)
            
            decisions.append(decision)
            log_probs.append(log_prob)
            probabilities.append(probs)
        
        return decisions, log_probs, probabilities
    
    def _compute_decision_hash(self, decision_type: str, decision_result: Any, features: torch.Tensor) -> str:
        """Compute decision hash value"""
        decision_str = f"{decision_type}_{decision_result}"
        if features is not None:
            features_data = features.cpu().numpy() if features.is_cuda else features.numpy()
            features_str = "_".join([f"{x:.3f}" for x in features_data[:5]])
            decision_str += f"_{features_str}"
        return hashlib.md5(decision_str.encode()).hexdigest()
    
    def create_decision_nodes_with_map_and_merge(self, 
                                               features_list: List[torch.Tensor],
                                               decision_type: str,
                                               decision_results: List[Any],
                                               log_probs: List[torch.Tensor],
                                               temperature: float,
                                               current_nodes: List[MultiBranchDecisionTreeNode]) -> List[MultiBranchDecisionTreeNode]:
        """MAP-based decision node creation with merge support."""
        new_nodes = []
        
        for i, (features, decision_result, log_prob, current_node) in enumerate(
            zip(features_list, decision_results, log_probs, current_nodes)
        ):
            node = MultiBranchDecisionTreeNode(
                node_id=f"{decision_type}_{time.time()}_{i}",
                features=features.detach().cpu(),
                decision_type=decision_type,
                decision_result=decision_result,
                log_prob=log_prob.detach().cpu(),
                temperature=temperature,
                path_id=current_node.path_id + f"_{decision_type}" if current_node.path_id else f"p{self.decision_tree.path_counter}_{decision_type}",
                branch_idx=current_node.branch_idx,
                session_idx=current_node.session_idx,
                parametric_memory_state=current_node.parametric_memory_state,
                non_parametric_memory_state=current_node.non_parametric_memory_state,
                decision_hash=self._compute_decision_hash(decision_type, decision_result, features),
                merge_count=current_node.merge_count
            )
            new_nodes.append(node)
        
        return new_nodes
    
    def execute_forgetting_decisions_with_map(self, param_memory, 
                                            features_list: List[torch.Tensor],
                                            conversation_texts: List[str],
                                            temperature: float,
                                            current_nodes: List[MultiBranchDecisionTreeNode]) -> Tuple[List[Dict[str, Any]], List[MultiBranchDecisionTreeNode]]:
        """MAP-based execution of forgetting decisions using stable_id for alignment."""
        forgetting_decisions = []
        new_decision_nodes = []
        
        if not current_nodes:
            return forgetting_decisions, new_decision_nodes
        
        min_len = min(len(features_list), len(conversation_texts), len(current_nodes))
        
        if min_len < len(current_nodes):
            print(f"⚠️  Warning: Forgetting adaptor input lists length mismatch. Using min length: {min_len}")
            current_nodes = current_nodes[:min_len]
            features_list = features_list[:min_len]
            conversation_texts = conversation_texts[:min_len]
        
        adapter_features_list = []
        non_param_features_list = []
        non_param_masks_list = []
        param_masks_list = []
        
        device = next(self.parameters()).device
        
        for i in range(min_len):
            node = current_nodes[i] if i < len(current_nodes) else None
            if node is None:
                adapter_features_list.append(torch.zeros(self.max_adapters, device=device))
                non_param_features_list.append(torch.zeros(self.max_non_param_memory, device=device))
                non_param_masks_list.append(torch.zeros(self.max_non_param_memory, device=device))
                param_masks_list.append(torch.zeros(self.max_adapters, device=device))
                continue
            
            param_memory_state = node.parametric_memory_state
            if param_memory_state is None:
                param_memory_state = param_memory
            
            if not IF_NO_PARAMETRIC:
                adapter_features = param_memory_state.get_adapter_features(
                    device, node.path_id, node.branch_idx
                )
                adapter_features_list.append(adapter_features)
            else:
                adapter_features_list.append(torch.zeros(self.max_adapters, device=device))
            
            if not IF_NO_NON_PARAMETRIC:
                non_parametric_memory = node.non_parametric_memory_state or []
                non_param_features = torch.zeros(self.max_non_param_memory, device=device)
                for j, mem in enumerate(non_parametric_memory[:self.max_non_param_memory]):
                    mem_str = str(mem) if isinstance(mem, str) else str(mem)
                    length_feature = min(len(mem_str) / 100.0, 1.0)
                    keyword_feature = 0.2 if any(kw in mem_str.lower() for kw in ["what", "how", "why"]) else 0.0
                    non_param_features[j] = length_feature + keyword_feature
                
                non_param_features_list.append(non_param_features)
                
                non_param_mask = torch.zeros(self.max_non_param_memory, device=device)
                for j in range(min(len(non_parametric_memory), self.max_non_param_memory)):
                    non_param_mask[j] = 1.0
            else:
                non_param_features_list.append(torch.zeros(self.max_non_param_memory, device=device))
                non_param_mask = torch.zeros(self.max_non_param_memory, device=device)
            
            if not IF_NO_PARAMETRIC:
                param_mask = param_memory_state.get_adapter_available_mask(device, node.path_id, node.branch_idx)
            else:
                param_mask = torch.zeros(self.max_adapters, device=device)
            
            non_param_masks_list.append(non_param_mask)
            param_masks_list.append(param_mask)
        
        level1_decisions, level1_log_probs, level1_probs = self.decide_level1_with_map(
            features_list, temperature, hard=True
        )
        
        if len(level1_decisions) != min_len:
            print(f"❌ Error: Forgetting level1 decisions mismatch: {len(level1_decisions)} decisions, {min_len} nodes. Creating default decisions.")
            for i in range(min_len):
                node = current_nodes[i] if i < len(current_nodes) else current_nodes[0]
                default_decision = create_default_forgetting_decision(node)
                forgetting_decisions.append(default_decision)
            
            return forgetting_decisions, current_nodes
        
        level1_nodes = self.create_decision_nodes_with_map_and_merge(
            features_list,
            "forgetting_level1",
            [d.item() > 0.5 for d in level1_decisions],
            level1_log_probs,
            temperature,
            current_nodes
        )
        
        if not level1_nodes:
            return forgetting_decisions, new_decision_nodes
        
        if level1_nodes:
            new_current_nodes = self.decision_tree.expand_all_current_with_map([[node] for node in level1_nodes])
        else:
            new_current_nodes = current_nodes
        
        if not new_current_nodes:
            return forgetting_decisions, new_decision_nodes
        
        for i in range(min_len):
            if i >= len(new_current_nodes) or i >= len(level1_decisions):
                break
                
            level1_node = new_current_nodes[i]
            level1_decision = level1_decisions[i]
            features = features_list[i] if i < len(features_list) else features_list[0]
            conversation_text = conversation_texts[i] if i < len(conversation_texts) else ""
            adapter_features = adapter_features_list[i] if i < len(adapter_features_list) else adapter_features_list[0]
            non_param_features = non_param_features_list[i] if i < len(non_param_features_list) else non_param_features_list[0]
            non_param_mask = non_param_masks_list[i] if i < len(non_param_masks_list) else non_param_masks_list[0]
            param_mask = param_masks_list[i] if i < len(param_masks_list) else param_masks_list[0]
            
            path_id = level1_node.path_id
            branch_idx = level1_node.branch_idx
            session_idx = level1_node.session_idx
            
            param_memory_state = level1_node.parametric_memory_state
            if param_memory_state is None:
                param_memory_state = param_memory
            
            non_parametric_memory = level1_node.non_parametric_memory_state or []
            
            decision = {
                "forget": level1_decision.item() > 0.5,
                "forget_type": None,
                "forget_non_param_idx": -1,
                "forget_param_idx": -1,
                "forget_param_name": None,
                "reason": "none",
                "log_prob": level1_log_probs[i].item() if i < len(level1_log_probs) else 0.0,
                "forget_probability": level1_probs[i].item() if i < len(level1_probs) else 0.0,
                "path_id": path_id,
                "branch_idx": branch_idx,
                "session_idx": session_idx,
                "non_param_memory_before": len(non_parametric_memory),
                "param_memory_before": len(param_memory_state.current_iteration_adapters) if param_memory_state else 0,
                "merge_count": level1_node.merge_count,
                "stable_id": level1_node.stable_id,
            }
            
            if decision["forget"]:
                decision["reason"] = "forget_decision_true"
                
                level2_decision, level2_log_prob, level2_probs = self.decide_level2_with_map(
                    [features], [adapter_features], [non_param_features], temperature, hard=True
                )
                
                decision["log_prob"] += level2_log_prob[0].item()
                
                if level2_decision[0].item() == 0 and not IF_NO_NON_PARAMETRIC:
                    decision["forget_type"] = "non_parametric"
                    
                    forget_idx, forget_log_prob, forget_probs = self.decide_non_param_selection_with_map(
                        [features], [non_param_features], [non_param_mask], temperature, hard=True
                    )
                    
                    decision["log_prob"] += forget_log_prob[0].item()
                    decision["forget_non_param_idx"] = forget_idx[0].item()
                    
                    if 0 <= forget_idx[0].item() < len(non_parametric_memory):
                        decision["forgotten_non_param"] = non_parametric_memory[forget_idx[0].item()]
                        non_parametric_memory.pop(forget_idx[0].item())
                        decision["reason"] = f"forget_non_parametric_idx_{forget_idx[0].item()}"
                        
                        non_param_nodes = self.create_decision_nodes_with_map_and_merge(
                            [features],
                            "forget_non_parametric",
                            [f"idx_{forget_idx[0].item()}"],
                            forget_log_prob,
                            temperature,
                            [level1_node]
                        )
                        
                        if non_param_nodes:
                            self.decision_tree.expand_all_current_with_map([[node] for node in non_param_nodes])
                            new_current_nodes = self.decision_tree.current_nodes
                    
                elif level2_decision[0].item() == 1 and not IF_NO_PARAMETRIC:
                    decision["forget_type"] = "parametric"
                    
                    forget_idx, forget_log_prob, forget_probs = self.decide_param_selection_with_map(
                        [features], [adapter_features], [param_mask], temperature, hard=True
                    )
                    
                    decision["log_prob"] += forget_log_prob[0].item()
                    decision["forget_param_idx"] = forget_idx[0].item()
                    
                    adapter_names = []
                    if path_id and path_id in param_memory_state.path_adapters:
                        adapter_names = param_memory_state.path_adapters[path_id]
                    else:
                        adapter_names = param_memory_state.current_iteration_adapters
                    
                    if 0 <= forget_idx[0].item() < len(adapter_names):
                        adapter_to_forget = adapter_names[forget_idx[0].item()]
                        decision["forget_param_name"] = adapter_to_forget
                        
                        if adapter_to_forget in param_memory_state.adapters:
                            param_memory_state.adapters[adapter_to_forget]["deleted"] = True
                            param_memory_state.adapters[adapter_to_forget]["deleted_time"] = time.time()
                            param_memory_state.adapters[adapter_to_forget]["deleted_by_path"] = path_id
                            param_memory_state.adapters[adapter_to_forget]["deleted_by_branch"] = branch_idx
                        
                        if adapter_to_forget in param_memory_state.current_iteration_adapters:
                            param_memory_state.current_iteration_adapters.remove(adapter_to_forget)
                        
                        if path_id and path_id in param_memory_state.path_adapters:
                            if adapter_to_forget in param_memory_state.path_adapters[path_id]:
                                param_memory_state.path_adapters[path_id].remove(adapter_to_forget)
                        
                        decision["reason"] = f"forget_parametric_{adapter_to_forget}"
                        
                        param_nodes = self.create_decision_nodes_with_map_and_merge(
                            [features],
                            "forget_parametric",
                            [adapter_to_forget],
                            forget_log_prob,
                            temperature,
                            [level1_node]
                        )
                        
                        if param_nodes:
                            self.decision_tree.expand_all_current_with_map([[node] for node in param_nodes])
                            new_current_nodes = self.decision_tree.current_nodes
                    
            else:
                decision["reason"] = "forget_decision_false"
            
            level1_node.parametric_memory_state = param_memory_state
            level1_node.non_parametric_memory_state = non_parametric_memory
            
            forgetting_decisions.append(decision)
            new_decision_nodes.append(level1_node)
        
        if len(forgetting_decisions) != len(new_current_nodes):
            print(f"⚠️  Warning: Forgetting decisions count mismatch after expansion: "
                  f"{len(forgetting_decisions)} decisions, {len(new_current_nodes)} nodes")
            
            _, aligned_forgetting_decisions = smart_align_nodes_and_decisions(
                new_current_nodes, forgetting_decisions, create_default_forgetting_decision, "forgetting"
            )
            forgetting_decisions = aligned_forgetting_decisions
        
        return forgetting_decisions, new_decision_nodes

    def compute_reinforce_loss(self, use_all_branches: bool = True) -> torch.Tensor:
        """Compute REINFORCE loss"""
        if use_all_branches:
            return self._compute_all_branches_loss()
        else:
            return self._compute_best_branch_loss()
    
    def _compute_best_branch_loss(self) -> torch.Tensor:
        """Compute loss for best branch"""
        if not self.best_branch_log_probs or not self.best_branch_normalized_rewards:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        
        min_len = min(len(self.best_branch_log_probs), len(self.best_branch_normalized_rewards))
        if min_len == 0:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        
        log_probs = self.best_branch_log_probs[:min_len]
        normalized_rewards = torch.tensor(
            self.best_branch_normalized_rewards[:min_len],
            device=next(self.parameters()).device
        )
        
        log_probs_stack = torch.stack(log_probs).to(normalized_rewards.device)
        loss = -(log_probs_stack * normalized_rewards).mean()
        
        return loss
    
    def _compute_all_branches_loss(self) -> torch.Tensor:
        """Compute loss for all branches"""
        if not self.all_branch_log_probs or not self.all_branch_relative_rewards:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        
        all_losses = []
        device = next(self.parameters()).device
        
        for log_probs_list, relative_rewards_list in zip(self.all_branch_log_probs, self.all_branch_relative_rewards):
            if not log_probs_list or not relative_rewards_list:
                continue
                
            if len(log_probs_list) != len(relative_rewards_list):
                min_len = min(len(log_probs_list), len(relative_rewards_list))
                log_probs_list = log_probs_list[:min_len]
                relative_rewards_list = relative_rewards_list[:min_len]
            
            valid_log_probs = []
            valid_relative_rewards = []
            
            for j in range(len(log_probs_list)):
                lp = log_probs_list[j]
                rr = relative_rewards_list[j]
                
                if (lp is not None and 
                    rr is not None and
                    hasattr(lp, 'numel') and 
                    lp.numel() > 0 and
                    not torch.isnan(lp).any() and
                    not torch.isinf(lp).any()):
                    
                    if lp.numel() == 1:
                        valid_log_probs.append(lp.view(-1))
                    else:
                        valid_log_probs.append(lp.mean().view(-1))
                    
                    valid_relative_rewards.append(rr)
            
            if valid_log_probs and valid_relative_rewards and len(valid_log_probs) > 0:
                try:
                    log_probs_tensors = []
                    for lp in valid_log_probs:
                        if lp.device != device:
                            lp = lp.to(device)
                        if len(lp.shape) == 0:
                            lp = lp.view(1)
                        elif len(lp.shape) > 1:
                            lp = lp.view(-1)
                        log_probs_tensors.append(lp)
                    
                    if all(lp.shape == torch.Size([1]) for lp in log_probs_tensors):
                        log_probs_stack = torch.stack(log_probs_tensors)
                    else:
                        scalar_log_probs = [lp.mean().view(1) for lp in log_probs_tensors]
                        log_probs_stack = torch.stack(scalar_log_probs)
                    
                    relative_rewards = torch.tensor(
                        valid_relative_rewards,
                        device=device,
                        dtype=torch.float32
                    )
                    
                    if log_probs_stack.shape[0] == relative_rewards.shape[0]:
                        if len(relative_rewards.shape) == 1:
                            relative_rewards = relative_rewards.view(-1, 1)
                        if len(log_probs_stack.shape) == 1:
                            log_probs_stack = log_probs_stack.view(-1, 1)
                        
                        branch_loss = -(log_probs_stack * relative_rewards).mean()
                        
                        if not torch.isnan(branch_loss) and not torch.isinf(branch_loss):
                            all_losses.append(branch_loss)
                        else:
                            print(f"⚠️  Warning: Invalid loss value (NaN/Inf) for branch")
                    else:
                        print(f"⚠️  Warning: Shape mismatch after stacking: log_probs {log_probs_stack.shape}, rewards {relative_rewards.shape}")
                except RuntimeError as e:
                    print(f"⚠️  Warning: Error stacking tensors: {e}")
                    continue
        
        if not all_losses:
            return torch.tensor(0.0, device=device)
        
        valid_losses = []
        for loss in all_losses:
            if loss is not None and not torch.isnan(loss) and not torch.isinf(loss):
                valid_losses.append(loss)
        
        if not valid_losses:
            return torch.tensor(0.0, device=device)
        
        return torch.stack(valid_losses).mean()

    def clear_decision_history(self):
        """Clear decision history"""
        self.best_branch_log_probs.clear()
        self.best_branch_normalized_rewards.clear()
        self.all_branch_log_probs.clear()
        self.all_branch_relative_rewards.clear()
        if self.decision_tree.root:
            self.decision_tree.current_nodes = [self.decision_tree.root]
        else:
            self.decision_tree = MultiBranchDecisionTree(
                max_branches=self.decision_tree.max_branches,
                merge_similarity_threshold=self.decision_tree.merge_similarity_threshold
            )
    
    def get_gradient_norm(self) -> float:
        """Get gradient norm"""
        total_norm = 0.0
        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5 if total_norm > 0 else 0.0

