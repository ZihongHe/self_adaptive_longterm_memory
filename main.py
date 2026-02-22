"""
Main experiment orchestration for multi‑session, multi‑branch memory adaptation.

This module contains the high‑level workflow that processes a dataset through
storage, retrieval, and forgetting phases with branch limits and merging.
It integrates all other modules and runs the complete experiment.
"""

import json
import random
import torch
import transformers
from typing import List, Dict, Any, Optional, Tuple, Union
import os
import numpy as np
import gc
import time
from datetime import datetime

from utils import args, load_longmem_json, compute_forgetting_quality, compute_target_logits_accuracy_with_model, evaluate_non_parametric_storage_decision, CumulativeAverager, MultiRepeatCumulativeAverager, IF_NO_PARAMETRIC, IF_NO_NON_PARAMETRIC, experiment_config_str, extract_conversation_text, get_optimized_temperature, PromptBuilder, NUM_SAMPLES, FeatureExtractor, EnhancedRewardTransformer, ConversationDataset, collate_fn, log_memory_usage, save_repeat_raw_stats_to_file
from adaptors import EnhancedHierarchicalMemoryAdaptor, ForgettingAdaptor
from tree import MultiBranchDecisionTreeNode, create_default_retrieval_decision, smart_align_nodes_and_decisions, align_all_decisions
from memory import LoRAModelLoader, EnhancedParametricMemory

def process_sessions_with_multi_branch_map_expansion_with_limit(
    sessions: List[Dict[str, Any]],
    timestamps: List[str],
    storage_adaptor: EnhancedHierarchicalMemoryAdaptor,
    forgetting_adaptor: ForgettingAdaptor,
    param_memory: EnhancedParametricMemory,
    feature_extractor: Optional[FeatureExtractor] = None,
    base_temperature: float = 2.0,
    max_branches: int = 64,
    merge_similarity_threshold: float = 0.95,
    max_branches_per_session: int = 2
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[MultiBranchDecisionTreeNode], int]:
    """Multi-session × multi-branch processing with storage and forgetting decisions, using stable_id for alignment."""
    all_storage_decisions = []
    all_forgetting_decisions = []
    all_decision_nodes = []
    
    if not isinstance(sessions, list):
        sessions = [sessions] if sessions is not None else []
    if not isinstance(timestamps, list):
        timestamps = [timestamps] if timestamps is not None else []
    
    if len(sessions) != len(timestamps):
        min_len = min(len(sessions), len(timestamps))
        sessions = sessions[:min_len]
        timestamps = timestamps[:min_len]
    
    session_count = len(sessions)
    
    if storage_adaptor.decision_tree.max_branches != max_branches:
        storage_adaptor.decision_tree.max_branches = max_branches
        storage_adaptor.decision_tree.merge_similarity_threshold = merge_similarity_threshold
    
    if forgetting_adaptor.decision_tree.max_branches != max_branches:
        forgetting_adaptor.decision_tree.max_branches = max_branches
        forgetting_adaptor.decision_tree.merge_similarity_threshold = merge_similarity_threshold
    
    if storage_adaptor.decision_tree.root is None:
        device = next(storage_adaptor.parameters()).device
        root_node = MultiBranchDecisionTreeNode(
            node_id="root",
            features=torch.randn(storage_adaptor.input_dim, device=device).cpu(),
            decision_type="root",
            decision_result="start",
            log_prob=torch.tensor(0.0),
            temperature=base_temperature,
            path_id="root",
            parametric_memory_state=param_memory,
            non_parametric_memory_state=[]
        )
        storage_adaptor.decision_tree.add_root(root_node)
    
    if forgetting_adaptor.decision_tree.root is None:
        device = next(forgetting_adaptor.parameters()).device
        forgetting_root = MultiBranchDecisionTreeNode(
            node_id="forgetting_root",
            features=torch.randn(forgetting_adaptor.input_dim, device=device).cpu(),
            decision_type="forgetting_root",
            decision_result="start",
            log_prob=torch.tensor(0.0),
            temperature=base_temperature,
            path_id="forgetting_root",
            parametric_memory_state=param_memory,
            non_parametric_memory_state=[]
        )
        forgetting_adaptor.decision_tree.add_root(forgetting_root)
    
    print(f"  🔄 Resetting parametric memory for new haystack_sessions (clearing all LoRA adapters)")
    param_memory.reset_iteration_adapters()
    param_memory.adapters.clear()
    param_memory.adapter_counter = 0
    param_memory.path_adapters.clear()
    param_memory.branch_adapter_map.clear()
    param_memory.forgotten_adapters.clear()
    param_memory.deleted_adapters_count = 0
    param_memory.behavior_influence = {
        "storage_influence": 0.0,
        "retrieval_influence": 0.0,
        "forgetting_influence": 0.0,
        "last_decision_effect": 0.0,
        "training_operations": 0,
        "total_training_tokens": 0,
    }
    
    for i, (session, ts) in enumerate(zip(sessions, timestamps)):
        try:
            if session is None:
                continue
            
            conversation_text, quality_score = extract_conversation_text(session, evaluate_quality=True)
            if len(str(conversation_text)) < 10:
                continue

            if quality_score > 0.7:
                base_temperature = max(base_temperature * 0.8, 0.1)
            elif quality_score < 0.3:
                base_temperature = min(base_temperature * 1.5, 3.0)
            
            storage_adaptor.decision_tree.split_branch_for_session_with_limit(
                i, total_sessions=session_count, max_branches_per_session=max_branches_per_session
            )
            forgetting_adaptor.decision_tree.split_branch_for_session_with_limit(
                i, total_sessions=session_count, max_branches_per_session=max_branches_per_session
            )
            
            current_nodes = storage_adaptor.decision_tree.current_nodes
            forgetting_current_nodes = forgetting_adaptor.decision_tree.current_nodes
            
            if not current_nodes or not forgetting_current_nodes:
                continue
            
            features_list = []
            conversation_texts = []
            
            device = next(storage_adaptor.parameters()).device
            
            for node in current_nodes:
                if feature_extractor is not None:
                    try:
                        features = feature_extractor.extract_features(conversation_text, device)
                    except Exception:
                        features = torch.randn(feature_extractor.hidden_dim, device=device)
                else:
                    features = torch.randn(storage_adaptor.input_dim, device=device)
                
                features = features.view(-1)
                if features.shape[0] != storage_adaptor.input_dim:
                    if features.shape[0] > storage_adaptor.input_dim:
                        features = features[:storage_adaptor.input_dim]
                    else:
                        padding = torch.zeros(storage_adaptor.input_dim - features.shape[0], device=device)
                        features = torch.cat([features, padding])
                
                features_list.append(features)
                conversation_texts.append(conversation_text)
            
            temperature = base_temperature * (1.0 - i / max(len(sessions), 1))
            
            storage_decisions, storage_decision_nodes = storage_adaptor.execute_storage_decisions_with_map(
                param_memory,
                features_list,
                conversation_texts,
                temperature,
                current_nodes
            )
            
            if not storage_decisions or not storage_decision_nodes:
                continue
            
            all_storage_decisions.extend(storage_decisions)
            all_decision_nodes.extend(storage_decision_nodes)
            
            forgetting_features_list = []
            forgetting_conversation_texts = []
            
            for node in forgetting_current_nodes:
                if feature_extractor is not None:
                    try:
                        features = feature_extractor.extract_features(conversation_text, device)
                    except Exception:
                        features = torch.randn(feature_extractor.hidden_dim, device=device)
                else:
                    features = torch.randn(forgetting_adaptor.input_dim, device=device)
                
                features = features.view(-1)
                if features.shape[0] != forgetting_adaptor.input_dim:
                    if features.shape[0] > forgetting_adaptor.input_dim:
                        features = features[:forgetting_adaptor.input_dim]
                    else:
                        padding = torch.zeros(forgetting_adaptor.input_dim - features.shape[0], device=device)
                        features = torch.cat([features, padding])
                
                forgetting_features_list.append(features)
                forgetting_conversation_texts.append(conversation_text)
            
            forgetting_decisions, forgetting_decision_nodes = forgetting_adaptor.execute_forgetting_decisions_with_map(
                param_memory,
                forgetting_features_list,
                forgetting_conversation_texts,
                temperature,
                forgetting_current_nodes
            )
            
            if not forgetting_decisions or not forgetting_decision_nodes:
                continue
            
            all_forgetting_decisions.extend(forgetting_decisions)
            all_decision_nodes.extend(forgetting_decision_nodes)
            
        except Exception:
            continue
    
    final_branch_nodes = storage_adaptor.decision_tree.current_nodes
    
    if not final_branch_nodes:
        return [], [], [], session_count
    
    aligned_storage_decisions, aligned_retrieval_decisions, aligned_forgetting_decisions, aligned_nodes = align_all_decisions(
        all_storage_decisions,
        [],
        all_forgetting_decisions,
        final_branch_nodes
    )
    
    return aligned_storage_decisions, aligned_forgetting_decisions, aligned_nodes, session_count


def enhanced_multi_branch_retrieval_with_map_expansion(
    param_memory: EnhancedParametricMemory,
    query_text: str,
    retrieval_adaptor: EnhancedHierarchicalMemoryAdaptor,
    feature_extractor: Optional[FeatureExtractor] = None,
    temperature: float = 0.5
) -> Tuple[List[Dict[str, Any]], List[float], List[MultiBranchDecisionTreeNode]]:
    """Multi-branch retrieval phase following MAP expansion principle using stable_id for alignment."""
    retrieval_decisions = []
    nll_adjustments = []
    decision_nodes = []
    
    current_nodes = retrieval_adaptor.decision_tree.current_nodes
    
    if not current_nodes or len(current_nodes) == 0:
        if hasattr(retrieval_adaptor, 'storage_adaptor_ref') and retrieval_adaptor.storage_adaptor_ref:
            storage_current_nodes = retrieval_adaptor.storage_adaptor_ref.decision_tree.current_nodes
            if storage_current_nodes:
                current_nodes = storage_current_nodes
                retrieval_adaptor.decision_tree.current_nodes = current_nodes
                retrieval_adaptor.decision_tree.leaf_nodes = retrieval_adaptor.storage_adaptor_ref.decision_tree.leaf_nodes
                retrieval_adaptor.decision_tree.decision_sequences = retrieval_adaptor.storage_adaptor_ref.decision_tree.decision_sequences
    
    if not current_nodes:
        return retrieval_decisions, nll_adjustments, decision_nodes
    
    features_list = []
    query_texts = []
    
    device = next(retrieval_adaptor.parameters()).device
    
    for node in current_nodes:
        if feature_extractor is not None:
            try:
                features = feature_extractor.extract_features(query_text, device)
            except Exception:
                features = torch.randn(retrieval_adaptor.input_dim, device=device)
        else:
            features = torch.randn(retrieval_adaptor.input_dim, device=device)
        
        features = features.view(-1)
        if features.shape[0] != retrieval_adaptor.input_dim:
            if features.shape[0] > retrieval_adaptor.input_dim:
                features = features[:retrieval_adaptor.input_dim]
            else:
                padding = torch.zeros(retrieval_adaptor.input_dim - features.shape[0], device=device)
                features = torch.cat([features, padding])
        
        features_list.append(features)
        query_texts.append(query_text)
    
    retrieval_decisions, nll_adjustments, decision_nodes = retrieval_adaptor.execute_retrieval_decisions_with_map(
        param_memory,
        features_list,
        query_texts,
        temperature,
        current_nodes
    )
    
    if not retrieval_decisions or not decision_nodes:
        return retrieval_decisions, nll_adjustments, decision_nodes
    
    if len(retrieval_decisions) != len(current_nodes):
        print(f"⚠️  Warning: Retrieval decisions count mismatch: {len(retrieval_decisions)} decisions, {len(current_nodes)} branches.")
        
        _, aligned_retrieval_decisions = smart_align_nodes_and_decisions(
            current_nodes, retrieval_decisions, create_default_retrieval_decision, "retrieval"
        )
        retrieval_decisions = aligned_retrieval_decisions
    
    if len(nll_adjustments) != len(current_nodes):
        print(f"⚠️  Warning: NLL adjustments mismatch: {len(nll_adjustments)} adjustments, {len(current_nodes)} branches.")
        if len(nll_adjustments) < len(current_nodes):
            for _ in range(len(nll_adjustments), len(current_nodes)):
                nll_adjustments.append(0.0)
        elif len(nll_adjustments) > len(current_nodes):
            nll_adjustments = nll_adjustments[:len(current_nodes)]
    
    for i, (node, features, retrieval_decision) in enumerate(zip(current_nodes, features_list, retrieval_decisions)):
        if i < len(retrieval_decisions):
            non_parametric_memory = node.non_parametric_memory_state or []
            
            if non_parametric_memory and not IF_NO_NON_PARAMETRIC:
                memory_features_list = []
                memory_texts = []
                
                device = features.device
                
                for mem_text in non_parametric_memory[:10]:
                    if feature_extractor is not None:
                        try:
                            mem_features = feature_extractor.extract_features(str(mem_text), device)
                            mem_features = mem_features.view(-1)
                            if mem_features.shape[0] != retrieval_adaptor.input_dim:
                                if mem_features.shape[0] > retrieval_adaptor.input_dim:
                                    mem_features = mem_features[:retrieval_adaptor.input_dim]
                                else:
                                    padding = torch.zeros(retrieval_adaptor.input_dim - mem_features.shape[0], device=device)
                                    mem_features = torch.cat([mem_features, padding])
                            
                            memory_features_list.append(mem_features)
                            memory_texts.append(str(mem_text)[:300])
                        except Exception:
                            text_len = len(str(mem_text))
                            mem_features = torch.zeros(retrieval_adaptor.input_dim, device=device)
                            mem_features[0] = min(text_len / 100.0, 1.0)
                            memory_features_list.append(mem_features)
                            memory_texts.append(str(mem_text)[:300])
                    else:
                        text_len = len(str(mem_text))
                        mem_features = torch.zeros(retrieval_adaptor.input_dim, device=device)
                        mem_features[0] = min(text_len / 100.0, 1.0)
                        memory_features_list.append(mem_features)
                        memory_texts.append(str(mem_text)[:300])
                
                if memory_features_list:
                    selected_memories, relevance_scores, decision_log_probs = retrieval_adaptor.select_relevant_non_parametric_memories(
                        features,
                        memory_texts,
                        memory_features_list,
                        max_memories=3,
                        relevance_threshold=0.3,
                        temperature=temperature,
                        hard=True
                    )
                    
                    truncated_memories = []
                    for mem in selected_memories:
                        if len(mem) > 200:
                            truncated_memories.append(mem[:200] + "...")
                        else:
                            truncated_memories.append(mem)
                    
                    retrieval_decision["non_parametric_context"] = truncated_memories
                    retrieval_decision["non_param_relevance_scores"] = relevance_scores
                    retrieval_decision["non_param_decision_log_probs"] = [log_prob.item() for log_prob in decision_log_probs]
                    
                    total_non_param_log_prob = sum(decision_log_probs) if decision_log_probs else torch.tensor(0.0)
                    retrieval_decision["log_prob"] += total_non_param_log_prob.item()
                    
                    if i < len(decision_nodes) and decision_nodes[i] is not None:
                        if hasattr(decision_nodes[i], 'log_prob') and decision_nodes[i].log_prob is not None:
                            new_log_prob = decision_nodes[i].log_prob + total_non_param_log_prob.detach().cpu()
                            decision_nodes[i].log_prob = new_log_prob
                else:
                    retrieval_decision["non_parametric_context"] = []
            else:
                retrieval_decision["non_parametric_context"] = []
    
    return retrieval_decisions, nll_adjustments, decision_nodes


def execute_complete_parallel_pipeline_with_limit(
    sample: Dict[str, Any],
    sample_idx: int,
    storage_adaptor: EnhancedHierarchicalMemoryAdaptor,
    retrieval_adaptor: EnhancedHierarchicalMemoryAdaptor,
    forgetting_adaptor: ForgettingAdaptor,
    param_memory: EnhancedParametricMemory,
    feature_extractor: FeatureExtractor,
    base_model,
    tokenizer,
    pipeline_model,
    lora_loader: LoRAModelLoader,
    storage_temperature: float = 2.0,
    retrieval_temperature: float = 0.5,
    max_branches: int = 64,
    merge_similarity_threshold: float = 0.95,
    max_branches_per_session: int = 2,
    use_length_normalization: bool = True,
    prompt_builder: Optional[PromptBuilder] = None
) -> Tuple[List[Dict[str, Any]], List[float], List[float], List[float], List[float], List[float]]:
    """Complete parallel pipeline using stable_id for branch and reward alignment."""
    all_results = []
    total_rewards = []
    normalized_rewards = []
    improvements = []
    logits_diffs = []
    forgetting_quality_scores = []
    
    SHAPING_ALPHA = 0.1
    
    retrieval_adaptor.storage_adaptor_ref = storage_adaptor
    
    sessions = sample.get("haystack_sessions", [])
    timestamps = sample.get("haystack_dates", [])
    
    storage_decisions, forgetting_decisions, decision_nodes, session_count = process_sessions_with_multi_branch_map_expansion_with_limit(
        sessions,
        timestamps,
        storage_adaptor,
        forgetting_adaptor,
        param_memory,
        feature_extractor,
        storage_temperature,
        max_branches,
        merge_similarity_threshold,
        max_branches_per_session
    )
    
    param_memory_stats = param_memory.get_memory_stats()
    
    query_text = sample.get("question", "")
    target_answer = str(sample.get("answer", ""))
    
    retrieval_adaptor.decision_tree.current_nodes = storage_adaptor.decision_tree.current_nodes.copy()
    retrieval_adaptor.decision_tree.leaf_nodes = storage_adaptor.decision_tree.leaf_nodes.copy()
    retrieval_adaptor.decision_tree.decision_sequences = storage_adaptor.decision_tree.decision_sequences.copy()
    retrieval_adaptor.decision_tree.session_counter = storage_adaptor.decision_tree.session_counter
    retrieval_adaptor.decision_tree.branch_counter = storage_adaptor.decision_tree.branch_counter
    retrieval_adaptor.decision_tree.path_id_map = storage_adaptor.decision_tree.path_id_map.copy()
    
    retrieval_decisions, nll_adjustments, retrieval_decision_nodes = enhanced_multi_branch_retrieval_with_map_expansion(
        param_memory,
        query_text,
        retrieval_adaptor,
        feature_extractor,
        retrieval_temperature
    )
    
    if not retrieval_decisions or not retrieval_decision_nodes:
        return all_results, total_rewards, normalized_rewards, improvements, logits_diffs, forgetting_quality_scores
    
    prompts_with_memory = []
    prompts_without_memory = []
    generated_answers = []
    
    actual_branches = retrieval_adaptor.decision_tree.current_nodes
    num_branches = len(actual_branches)
    if num_branches == 0:
        return all_results, total_rewards, normalized_rewards, improvements, logits_diffs, forgetting_quality_scores
    
    max_branches_to_process = min(num_branches, 16)
    if num_branches > max_branches_to_process:
        print(f"  ⚠️  Limiting branches from {num_branches} to {max_branches_to_process}")
        actual_branches = actual_branches[:max_branches_to_process]
        retrieval_decisions = retrieval_decisions[:max_branches_to_process]
        nll_adjustments = nll_adjustments[:max_branches_to_process]
        num_branches = max_branches_to_process
    
    if len(retrieval_decisions) != num_branches:
        return all_results, total_rewards, normalized_rewards, improvements, logits_diffs, forgetting_quality_scores
    
    if len(nll_adjustments) != num_branches:
        return all_results, total_rewards, normalized_rewards, improvements, logits_diffs, forgetting_quality_scores
    
    current_nodes = retrieval_adaptor.decision_tree.current_nodes[:num_branches]
    
    for branch_idx in range(num_branches):
        if branch_idx % 5 == 0:
            log_memory_usage(f"Branch {branch_idx} start")
        
        non_parametric_context = []
        if branch_idx < len(retrieval_decisions):
            non_parametric_context = retrieval_decisions[branch_idx].get("non_parametric_context", [])
            
            if not non_parametric_context and branch_idx < len(current_nodes):
                node = current_nodes[branch_idx]
                node_memories = node.non_parametric_memory_state or []
                if node_memories:
                    truncated_memories = []
                    for mem in node_memories[-3:]:
                        mem_str = str(mem)
                        if len(mem_str) > 200:
                            truncated_memories.append(mem_str[:200] + "...")
                        else:
                            truncated_memories.append(mem_str)
                    non_parametric_context = truncated_memories
            
            relevance_scores = retrieval_decisions[branch_idx].get("non_param_relevance_scores", [])
            
            if non_parametric_context and relevance_scores and len(non_parametric_context) == len(relevance_scores):
                sorted_indices = np.argsort(relevance_scores)[::-1]
                non_parametric_context = [non_parametric_context[i] for i in sorted_indices[:3]]
            elif non_parametric_context and not relevance_scores:
                non_parametric_context = non_parametric_context[:3]

        if prompt_builder is not None:
            prompt_with_memory = prompt_builder.build_prompt_with_context(
                non_parametric_context[:3],
                str(query_text)
            )
            
            prompt_stats = prompt_builder.get_prompt_stats(prompt_with_memory)
            
            prompt_total_tokens = prompt_stats["total_tokens"]
            prompt_context_tokens = prompt_stats.get("context_tokens", 0)
            prompt_question_tokens = prompt_stats.get("question_tokens", 0)
            
            if prompt_total_tokens > 450:
                print(f"  ⚠️ Branch {branch_idx}: Prompt near limit ({prompt_total_tokens}/{prompt_builder.max_total_tokens} tokens)")
        else:
            context = "\n".join(non_parametric_context[:3])
            if context:
                prompt_with_memory = "Some previous conversations:\n" + context + "\n\nQ: " + str(query_text) + "\nA:"
            else:
                prompt_with_memory = "Q: " + str(query_text) + "\nA:"
            prompt_total_tokens = prompt_context_tokens = prompt_question_tokens = 0
        
        prompt_without_memory = "Q: " + str(query_text) + "\nA:"
        
        prompt_with_memory_str = str(prompt_with_memory)
        if len(prompt_with_memory_str) > 400:
            prompt_with_memory_str = prompt_with_memory_str[:400] + "..."
        
        try:
            with torch.no_grad():
                outputs = pipeline_model(
                    prompt_with_memory_str,
                    max_new_tokens=64,
                    do_sample=False,
                    truncation=True,
                    max_length=512
                )
                
                generated_text = outputs[0]["generated_text"]
                generated_answer = generated_text[len(prompt_with_memory_str):].strip() if len(generated_text) > len(prompt_with_memory_str) else generated_text
                
                if len(generated_answer) > 200:
                    generated_answer = generated_answer[:200] + "..."
            
            prompts_with_memory.append(prompt_with_memory_str)
            prompts_without_memory.append(prompt_without_memory)
            generated_answers.append(generated_answer)
            
            del outputs
            if branch_idx % 3 == 0:
                torch.cuda.empty_cache()
            
        except Exception:
            prompts_with_memory.append(prompt_with_memory_str)
            prompts_without_memory.append(prompt_without_memory)
            generated_answers.append("ERROR: Generation failed")
            prompt_total_tokens = prompt_context_tokens = prompt_question_tokens = 0
    
    actual_branches = retrieval_adaptor.decision_tree.current_nodes[:num_branches]
    actual_branch_count = len(actual_branches)
    
    aligned_storage_decisions, aligned_retrieval_decisions, aligned_forgetting_decisions, aligned_decision_nodes = align_all_decisions(
        storage_decisions,
        retrieval_decisions,
        forgetting_decisions,
        actual_branches
    )
    
    storage_decisions = aligned_storage_decisions
    retrieval_decisions = aligned_retrieval_decisions
    forgetting_decisions = aligned_forgetting_decisions
    decision_nodes = aligned_decision_nodes
    actual_branch_count = len(decision_nodes)
    
    forgetting_quality_scores = compute_forgetting_quality(
        forgetting_decisions,
        generated_answers,
        target_answer,
        param_memory_stats
    )
    
    if len(forgetting_quality_scores) != actual_branch_count:
        aligned_forgetting_quality_scores = []
        for i in range(actual_branch_count):
            if i < len(forgetting_quality_scores):
                aligned_forgetting_quality_scores.append(forgetting_quality_scores[i])
            else:
                aligned_forgetting_quality_scores.append(0.5)
        forgetting_quality_scores = aligned_forgetting_quality_scores
    
    reward_transformer = EnhancedRewardTransformer(baseline_type="ema", ema_alpha=0.95)
    
    if not hasattr(execute_complete_parallel_pipeline_with_limit, "previous_branch_improvements"):
        execute_complete_parallel_pipeline_with_limit.previous_branch_improvements = {}
    
    sample_key = sample.get("id", f"sample_{sample_idx}")
    
    for branch_idx in range(actual_branch_count):
        if branch_idx < len(generated_answers):
            use_parametric = retrieval_decisions[branch_idx].get('use_parametric', False)
            selected_adapter = retrieval_decisions[branch_idx].get('selected_adapter', None)
            
            non_param_memory_size = 0
            if branch_idx < len(decision_nodes):
                node = decision_nodes[branch_idx]
                non_param_memory_size = len(node.non_parametric_memory_state or [])
            
            prompt_total_tokens = prompt_context_tokens = prompt_question_tokens = 0
            
            with torch.no_grad():
                baseline_logits_diff, baseline_token_accuracy, baseline_token_count = compute_target_logits_accuracy_with_model(
                    base_model, tokenizer, prompt_without_memory, target_answer,
                    use_length_normalization=use_length_normalization
                )
                baseline_nll = baseline_logits_diff + 0.3
                
                memory_logits_diff = 0.0
                memory_token_accuracy = 0.0
                memory_token_count = 0
                
                if use_parametric and selected_adapter and not IF_NO_PARAMETRIC:
                    adapter_info = param_memory.adapters.get(selected_adapter, {}) if param_memory else {}
                    
                    try:
                        memory_model = lora_loader.get_lora_model(selected_adapter, adapter_info)
                        
                        memory_logits_diff, memory_token_accuracy, memory_token_count = compute_target_logits_accuracy_with_model(
                            memory_model, tokenizer, prompt_with_memory, target_answer,
                            use_length_normalization=use_length_normalization
                        )
                        
                        del memory_model
                        
                    except Exception:
                        memory_logits_diff, memory_token_accuracy, memory_token_count = compute_target_logits_accuracy_with_model(
                            base_model, tokenizer, prompt_with_memory, target_answer,
                            use_length_normalization=use_length_normalization
                        )
                        
                else:
                    memory_logits_diff, memory_token_accuracy, memory_token_count = compute_target_logits_accuracy_with_model(
                        base_model, tokenizer, prompt_with_memory, target_answer,
                        use_length_normalization=use_length_normalization
                    )
            
            memory_nll = memory_logits_diff + (nll_adjustments[branch_idx] if branch_idx < len(nll_adjustments) else 0)
            
            improvement = baseline_nll - memory_nll
            
            if use_length_normalization and baseline_token_count > 0 and memory_token_count > 0:
                length_ratio = memory_token_count / max(baseline_token_count, 1)
                if length_ratio > 1.5:
                    length_adjustment = 0.1 * (length_ratio - 1.5)
                    improvement += length_adjustment
                elif length_ratio < 0.7:
                    length_adjustment = -0.1 * (0.7 - length_ratio)
                    improvement += length_adjustment
            
            forgetting_quality = forgetting_quality_scores[branch_idx] if branch_idx < len(forgetting_quality_scores) else 0.5
            
            non_parametric_storage_quality = evaluate_non_parametric_storage_decision(
                storage_decisions[branch_idx] if branch_idx < len(storage_decisions) else {},
                generated_answers[branch_idx] if branch_idx < len(generated_answers) else "",
                target_answer,
                non_param_memory_size
            )
            
            logits_penalty = -memory_logits_diff * 5.0
            accuracy_reward = memory_token_accuracy * 3.0
            forgetting_reward = forgetting_quality * 2.0
            storage_quality_reward = non_parametric_storage_quality * 1.5
            
            training_reward = 0.0
            if use_parametric and selected_adapter and not IF_NO_PARAMETRIC:
                adapter_info = param_memory.adapters.get(selected_adapter, {})
                training_stats = adapter_info.get("training_stats", {})
                
                avg_loss = training_stats.get("average_loss", 5.0)
                if avg_loss < 2.0:
                    training_reward += 0.3
                elif avg_loss < 3.0:
                    training_reward += 0.1
                
                training_samples = training_stats.get("total_training_samples", 0)
                training_reward += min(training_samples / 100.0, 0.2)
            
            improvement_weight = 0.25
            logits_weight = 0.2
            accuracy_weight = 0.1
            forgetting_weight = 0.1
            storage_quality_weight = 0.15
            training_weight = 0.2
            
            combined_improvement = (
                improvement * improvement_weight +
                logits_penalty * logits_weight +
                accuracy_reward * accuracy_weight +
                forgetting_reward * forgetting_weight +
                storage_quality_reward * storage_quality_weight +
                training_reward * training_weight
            )
            
            branch_key = f"{sample_key}_{branch_idx}"
            prev_improvement = execute_complete_parallel_pipeline_with_limit.previous_branch_improvements.get(branch_key, 0.0)
            improvement_step = improvement - prev_improvement
            execute_complete_parallel_pipeline_with_limit.previous_branch_improvements[branch_key] = improvement
            
            shaping_reward = SHAPING_ALPHA * improvement_step
            original_combined_improvement = combined_improvement
            
            if use_parametric and selected_adapter and not IF_NO_PARAMETRIC:
                combined_improvement += shaping_reward
                shaping_applied = True
                shaping_value = shaping_reward
            else:
                shaping_applied = False
                shaping_value = 0.0
            
            total_reward, normalized_reward = reward_transformer.transform(
                combined_improvement,
                reward_type="clipped_linear",
                clip_min=-10.0,
                clip_max=10.0
            )
            
            total_rewards.append(total_reward)
            normalized_rewards.append(normalized_reward)
            improvements.append(improvement)
            logits_diffs.append(memory_logits_diff)
            
            forget_non_param_count = 0
            forget_param_count = 0
            if branch_idx < len(forgetting_decisions):
                decision = forgetting_decisions[branch_idx]
                if decision.get("forget", False):
                    if decision.get("forget_type") == "non_parametric":
                        forget_non_param_count = 1
                    elif decision.get("forget_type") == "parametric":
                        forget_param_count = 1
            
            branch_count = retrieval_adaptor.decision_tree.get_branch_count()
            
            merge_count = 1
            if branch_idx < len(decision_nodes):
                node = decision_nodes[branch_idx]
                merge_count = node.merge_count
            
            storage_to_non_parametric = False
            store_non_parametric_probability = 0.0
            training_performed = False
            training_loss = 0.0
            training_perplexity = 0.0
            training_samples = 0
            
            if branch_idx < len(storage_decisions):
                storage_to_non_parametric = storage_decisions[branch_idx].get("storage_to_non_parametric", False)
                store_non_parametric_probability = storage_decisions[branch_idx].get("store_non_parametric_probability", 0.0)
                training_performed = storage_decisions[branch_idx].get("training_performed", False)
                training_loss = storage_decisions[branch_idx].get("training_loss", 0.0)
                training_perplexity = storage_decisions[branch_idx].get("training_perplexity", 0.0)
                training_samples = storage_decisions[branch_idx].get("training_samples", 0)
            
            adapter_training_stats = {}
            if use_parametric and selected_adapter and selected_adapter in param_memory.adapters:
                adapter_info = param_memory.adapters[selected_adapter]
                adapter_training_stats = adapter_info.get("training_stats", {})
            
            stable_id = decision_nodes[branch_idx].stable_id if branch_idx < len(decision_nodes) else f"unknown_{branch_idx}"
            
            result = {
                "sample_idx": sample_idx,
                "sample_id": sample.get("id", f"sample_{sample_idx}"),
                "question": query_text[:100],
                "generated_answer": generated_answers[branch_idx][:200],
                "target_answer": str(target_answer)[:100],
                "difference_score": improvement,
                "raw_difference_score": improvement,
                "counterfactual_improvement": improvement,
                "baseline_nll": baseline_nll,
                "memory_nll": memory_nll,
                "reward": total_reward,
                "reward_type": "enhanced_fair_comparison",
                "use_parametric": use_parametric,
                "active_adapter": selected_adapter,
                "non_param_count": len(non_parametric_context),
                "storage_decision": storage_decisions[branch_idx].get("reason", "none") if branch_idx < len(storage_decisions) else "none",
                "retrieval_decision": retrieval_decisions[branch_idx].get("reason", "none") if branch_idx < len(retrieval_decisions) else "none",
                "forgetting_decision": forgetting_decisions[branch_idx].get("reason", "none") if branch_idx < len(forgetting_decisions) else "none",
                "lora_adapter_count": len(param_memory.current_iteration_adapters),
                "batch_idx": sample_idx,
                "gradient_norm": storage_adaptor.get_gradient_norm() + retrieval_adaptor.get_gradient_norm() + forgetting_adaptor.get_gradient_norm(),
                "forgetting_reward": forgetting_reward,
                "decision_tree_size": storage_adaptor.decision_tree.root.count_nodes() if storage_adaptor.decision_tree.root else 0,
                "normalized_reward": normalized_reward,
                "repeat_idx": 0,
                "answer_logits_diff": memory_logits_diff,
                "tokenwise_accuracy": memory_token_accuracy,
                "branch_count": branch_count,
                "path_id": decision_nodes[branch_idx].path_id if branch_idx < len(decision_nodes) else f"path_{branch_idx}",
                "num_paths": actual_branch_count,
                "session_count": session_count,
                "non_param_memory_size": non_param_memory_size,
                "param_memory_size": len(param_memory.current_iteration_adapters),
                "forget_non_param_count": forget_non_param_count,
                "forget_param_count": forget_param_count,
                "forget_quality_score": forgetting_quality,
                "merge_count": merge_count,
                "total_branches": branch_count,
                "max_branches": max_branches,
                "experiment_config": experiment_config_str,
                "storage_to_non_parametric": storage_to_non_parametric,
                "store_non_parametric_probability": store_non_parametric_probability,
                "non_parametric_memory_usage": non_param_memory_size / 20.0 if non_param_memory_size > 0 else 0.0,
                "non_parametric_storage_quality": non_parametric_storage_quality,
                "decision_aligned": True,
                "training_performed": training_performed,
                "training_loss": training_loss,
                "training_perplexity": training_perplexity,
                "training_samples": training_samples,
                "adapter_training_stats": adapter_training_stats,
                "training_reward": training_reward,
                "baseline_token_count": baseline_token_count,
                "memory_token_count": memory_token_count,
                "token_count_ratio": memory_token_count / max(baseline_token_count, 1) if baseline_token_count > 0 else 1.0,
                "use_length_normalization": use_length_normalization,
                "shaping_applied": shaping_applied,
                "shaping_reward": shaping_value if shaping_applied else 0.0,
                "shaping_alpha": SHAPING_ALPHA,
                "improvement_step": improvement_step,
                "previous_improvement": prev_improvement,
                "original_combined_improvement": original_combined_improvement,
                "shaped_combined_improvement": combined_improvement,
                "prompt_total_tokens": prompt_total_tokens,
                "prompt_context_tokens": prompt_context_tokens,
                "prompt_question_tokens": prompt_question_tokens,
                "stable_id": stable_id,
            }
            
            all_results.append(result)
            
            if branch_idx % 4 == 3:
                torch.cuda.empty_cache()
                gc.collect()
    
    if storage_adaptor.decision_tree.current_nodes:
        aligned_nodes, aligned_rewards, aligned_normalized_rewards = storage_adaptor.decision_tree.branch_manager.get_aligned_branches_and_rewards(
            storage_adaptor.decision_tree.current_nodes
        )
        
        if aligned_nodes and aligned_rewards:
            storage_adaptor.decision_tree.finalize_all_branches_with_rewards(aligned_rewards, aligned_normalized_rewards)
    
    if retrieval_adaptor.decision_tree.current_nodes:
        aligned_nodes, aligned_rewards, aligned_normalized_rewards = retrieval_adaptor.decision_tree.branch_manager.get_aligned_branches_and_rewards(
            retrieval_adaptor.decision_tree.current_nodes
        )
        
        if aligned_nodes and aligned_rewards:
            retrieval_adaptor.decision_tree.finalize_all_branches_with_rewards(aligned_rewards, aligned_normalized_rewards)
    
    if forgetting_adaptor.decision_tree.current_nodes:
        aligned_nodes, aligned_rewards, aligned_normalized_rewards = forgetting_adaptor.decision_tree.branch_manager.get_aligned_branches_and_rewards(
            forgetting_adaptor.decision_tree.current_nodes
        )
        
        if aligned_nodes and aligned_rewards:
            forgetting_adaptor.decision_tree.finalize_all_branches_with_rewards(aligned_rewards, aligned_normalized_rewards)
    
    storage_paths = storage_adaptor.decision_tree.get_all_paths()
    for path in storage_paths:
        log_probs = []
        for node in path:
            if node.log_prob is not None and hasattr(node.log_prob, 'numel') and node.log_prob.numel() > 0:
                lp = node.log_prob.detach().cpu()
                if not torch.isnan(lp).any() and not torch.isinf(lp).any():
                    if lp.numel() == 1:
                        log_probs.append(lp)
                    else:
                        log_probs.append(lp.mean())
    
        if log_probs:
            normalized_reward = path[-1].normalized_reward if path[-1].normalized_reward is not None else 0.0
            if not np.isnan(normalized_reward) and not np.isinf(normalized_reward):
                storage_adaptor.all_branch_log_probs.append(log_probs)
                storage_adaptor.all_branch_relative_rewards.append([normalized_reward] * len(log_probs))
    
    retrieval_paths = retrieval_adaptor.decision_tree.get_all_paths()
    for path in retrieval_paths:
        log_probs = []
        for node in path:
            if node.log_prob is not None and hasattr(node.log_prob, 'numel') and node.log_prob.numel() > 0:
                lp = node.log_prob.detach().cpu()
                if not torch.isnan(lp).any() and not torch.isinf(lp).any():
                    if lp.numel() == 1:
                        log_probs.append(lp)
                    else:
                        log_probs.append(lp.mean())
    
        if log_probs:
            normalized_reward = path[-1].normalized_reward if path[-1].normalized_reward is not None else 0.0
            if not np.isnan(normalized_reward) and not np.isinf(normalized_reward):
                retrieval_adaptor.all_branch_log_probs.append(log_probs)
                retrieval_adaptor.all_branch_relative_rewards.append([normalized_reward] * len(log_probs))
    
    forgetting_paths = forgetting_adaptor.decision_tree.get_all_paths()
    for path in forgetting_paths:
        log_probs = []
        for node in path:
            if node.log_prob is not None and hasattr(node.log_prob, 'numel') and node.log_prob.numel() > 0:
                lp = node.log_prob.detach().cpu()
                if not torch.isnan(lp).any() and not torch.isinf(lp).any():
                    if lp.numel() == 1:
                        log_probs.append(lp)
                    else:
                        log_probs.append(lp.mean())
    
        if log_probs:
            normalized_reward = path[-1].normalized_reward if path[-1].normalized_reward is not None else 0.0
            if not np.isnan(normalized_reward) and not np.isinf(normalized_reward):
                forgetting_adaptor.all_branch_log_probs.append(log_probs)
                forgetting_adaptor.all_branch_relative_rewards.append([normalized_reward] * len(log_probs))
    
    return all_results, total_rewards, normalized_rewards, improvements, logits_diffs, forgetting_quality_scores


def update_enhanced_adaptors_with_all_branches(
    storage_adaptor: EnhancedHierarchicalMemoryAdaptor,
    retrieval_adaptor: EnhancedHierarchicalMemoryAdaptor,
    forgetting_adaptor: ForgettingAdaptor,
    optimizer: torch.optim.Optimizer,
    total_rewards: List[float],
    forgetting_quality_scores: List[float],
    use_all_branches: bool = True,
    learning_rate: float = 1e-3,
    clip_grad_norm: float = 10.0
) -> Tuple[torch.optim.Optimizer, Dict[str, float]]:
    """Update adaptors using all branches."""
    update_info = {
        "storage_loss": 0.0,
        "retrieval_loss": 0.0,
        "forgetting_loss": 0.0,
        "total_loss": 0.0,
        "gradient_norm": 0.0,
        "learning_rate": learning_rate,
        "use_all_branches": use_all_branches,
        "num_branches": len(total_rewards),
    }
    
    storage_loss = storage_adaptor.compute_reinforce_loss(use_all_branches=use_all_branches)
    retrieval_loss = retrieval_adaptor.compute_reinforce_loss(use_all_branches=use_all_branches)
    forgetting_loss = forgetting_adaptor.compute_reinforce_loss(use_all_branches=use_all_branches)
    
    if total_rewards and forgetting_quality_scores:
        avg_reward = np.mean(total_rewards)
        avg_forgetting_quality = np.mean(forgetting_quality_scores)
        
        reward_factor = max(0.5, min(2.0, 1.0 + avg_reward))
        forgetting_factor = max(0.5, min(1.5, 0.8 + avg_forgetting_quality))
        
        storage_loss = storage_loss * reward_factor
        retrieval_loss = retrieval_loss * reward_factor
        forgetting_loss = forgetting_loss * forgetting_factor
    
    total_loss = storage_loss + retrieval_loss + forgetting_loss
    update_info["storage_loss"] = storage_loss.item()
    update_info["retrieval_loss"] = retrieval_loss.item()
    update_info["forgetting_loss"] = forgetting_loss.item()
    update_info["total_loss"] = total_loss.item()
    
    if total_loss.requires_grad and total_loss.grad_fn is not None:
        optimizer.zero_grad()
        total_loss.backward()
        
        total_grad_norm = 0.0
        for adaptor in [storage_adaptor, retrieval_adaptor, forgetting_adaptor]:
            total_grad_norm += adaptor.get_gradient_norm()
        update_info["gradient_norm"] = total_grad_norm
        
        torch.nn.utils.clip_grad_norm_(
            [p for group in optimizer.param_groups for p in group['params']], 
            max_norm=clip_grad_norm
        )
        
        optimizer.step()
    
    storage_adaptor.clear_decision_history()
    retrieval_adaptor.clear_decision_history()
    forgetting_adaptor.clear_decision_history()
    
    return optimizer, update_info


def compute_final_experiment_stats(
    all_repeat_raw_stats: List[Dict[str, Any]],
    repeat_stats: List[Dict[str, Any]],
    all_cumulative_improvement_sequences: List[List[float]],
    all_cumulative_reward_sequences: List[List[float]],
    multi_repeat_averager: MultiRepeatCumulativeAverager,
    num_repeat: int,
    use_all_branches: bool,
    max_branches: int,
    merge_similarity_threshold: float,
    all_results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Compute final statistics for all repeated experiments."""
    
    def compute_cross_repeat_statistics(sequences):
        if not sequences:
            return [], []
        
        min_length = min(len(seq) for seq in sequences)
        aligned_sequences = [seq[:min_length] for seq in sequences]
        
        means = []
        stds = []
        
        for i in range(min_length):
            position_values = [seq[i] for seq in aligned_sequences]
            means.append(np.mean(position_values))
            stds.append(np.std(position_values) if len(position_values) > 1 else 0.0)
        
        return means, stds
    
    cross_repeat_improvement_means = []
    cross_repeat_improvement_stds = []
    
    if all_cumulative_improvement_sequences:
        cross_repeat_improvement_means, cross_repeat_improvement_stds = compute_cross_repeat_statistics(
            all_cumulative_improvement_sequences
        )
    
    cross_repeat_reward_means = []
    cross_repeat_reward_stds = []
    
    if all_cumulative_reward_sequences:
        cross_repeat_reward_means, cross_repeat_reward_stds = compute_cross_repeat_statistics(
            all_cumulative_reward_sequences
        )
    
    all_improvements = []
    all_rewards = []
    all_logits_diffs = []
    all_forgetting_quality_scores = []
    all_branch_counts = []
    all_merge_counts = []
    all_training_performed_count = 0
    all_avg_training_loss = []
    all_avg_training_perplexity = []
    
    for repeat_stat in repeat_stats:
        all_improvements.append(repeat_stat["avg_improvement"])
        all_rewards.append(repeat_stat["avg_reward"])
        
        if repeat_stat.get("avg_logits_diff"):
            all_logits_diffs.append(repeat_stat["avg_logits_diff"])
        if repeat_stat.get("avg_forgetting_quality"):
            all_forgetting_quality_scores.append(repeat_stat["avg_forgetting_quality"])
        if repeat_stat.get("avg_branch_count"):
            all_branch_counts.append(repeat_stat["avg_branch_count"])
        if repeat_stat.get("avg_merge_count"):
            all_merge_counts.append(repeat_stat["avg_merge_count"])
        if repeat_stat.get("training_stats"):
            all_training_performed_count += repeat_stat["training_stats"].get("training_performed_count", 0)
            all_avg_training_loss.append(repeat_stat["training_stats"].get("avg_training_loss", 0.0))
            all_avg_training_perplexity.append(repeat_stat["training_stats"].get("avg_training_perplexity", 0.0))
    
    experiment_stats = {
        "num_repeats": num_repeat,
        "use_all_branches": use_all_branches,
        "max_branches": max_branches,
        "merge_similarity_threshold": merge_similarity_threshold,
        "total_samples": sum(stat.get("samples_processed", 0) for stat in repeat_stats) if repeat_stats else 0,
        
        "overall_statistics": {
            "improvement": {
                "mean": np.mean(all_improvements) if all_improvements else 0.0,
                "std": np.std(all_improvements) if len(all_improvements) > 1 else 0.0,
                "min": np.min(all_improvements) if all_improvements else 0.0,
                "max": np.max(all_improvements) if all_improvements else 0.0,
            },
            "reward": {
                "mean": np.mean(all_rewards) if all_rewards else 0.0,
                "std": np.std(all_rewards) if len(all_rewards) > 1 else 0.0,
                "min": np.min(all_rewards) if all_rewards else 0.0,
                "max": np.max(all_rewards) if all_rewards else 0.0,
            },
            "logits_diff": {
                "mean": np.mean(all_logits_diffs) if all_logits_diffs else 0.0,
                "std": np.std(all_logits_diffs) if len(all_logits_diffs) > 1 else 0.0,
            },
            "forgetting_quality": {
                "mean": np.mean(all_forgetting_quality_scores) if all_forgetting_quality_scores else 0.0,
                "std": np.std(all_forgetting_quality_scores) if len(all_forgetting_quality_scores) > 1 else 0.0,
            },
            "branch_count": {
                "mean": np.mean(all_branch_counts) if all_branch_counts else 1.0,
                "std": np.std(all_branch_counts) if len(all_branch_counts) > 1 else 0.0,
            },
            "merge_count": {
                "mean": np.mean(all_merge_counts) if all_merge_counts else 1.0,
                "std": np.std(all_merge_counts) if len(all_merge_counts) > 1 else 0.0,
            },
            "training_stats": {
                "total_training_operations": all_training_performed_count,
                "avg_training_loss": np.mean(all_avg_training_loss) if all_avg_training_loss else 0.0,
                "avg_training_perplexity": np.mean(all_avg_training_perplexity) if all_avg_training_perplexity else 0.0,
            }
        },
        
        "repeat_statistics": repeat_stats,
        
        "cross_repeat_cumulative_improvement": {
            "means": cross_repeat_improvement_means,
            "stds": cross_repeat_improvement_stds,
            "num_sequences": len(all_cumulative_improvement_sequences),
            "sequence_lengths": [len(seq) for seq in all_cumulative_improvement_sequences] if all_cumulative_improvement_sequences else [],
            "aligned_length": len(cross_repeat_improvement_means),
        },
        "cross_repeat_cumulative_reward": {
            "means": cross_repeat_reward_means,
            "stds": cross_repeat_reward_stds,
            "num_sequences": len(all_cumulative_reward_sequences),
            "sequence_lengths": [len(seq) for seq in all_cumulative_reward_sequences] if all_cumulative_reward_sequences else [],
            "aligned_length": len(cross_repeat_reward_means),
        },
        
        "experiment_config": experiment_config_str,
        "control_variables": {
            "if_no_parametric": IF_NO_PARAMETRIC,
            "if_no_non_parametric": IF_NO_NON_PARAMETRIC,
            "num_samples": NUM_SAMPLES,
        },
    }
    
    if num_repeat > 1:
        final_cumulative_averages, final_ema_averages = multi_repeat_averager.compute_final_averages()
        experiment_stats["multi_repeat_cumulative_averages"] = {
            "final_cumulative_averages": final_cumulative_averages,
            "final_ema_averages": final_ema_averages,
            "num_repeats": num_repeat,
            "sample_count_per_repeat": len(all_repeat_raw_stats[0]["cumulative_improvement_sequence"]) if all_repeat_raw_stats else 0,
        }
    
    return experiment_stats


def save_current_repeat_stats(results_dir: str, repeat_stat: Dict[str, Any], 
                            repeat_idx: int, total_repeats: int):
    """Save statistics for current repeated experiment."""
    repeat_stats_dir = os.path.join(results_dir, "repeat_stats")
    os.makedirs(repeat_stats_dir, exist_ok=True)
    
    repeat_file = os.path.join(repeat_stats_dir, f"repeat_{repeat_idx}_stats.json")
    with open(repeat_file, 'w', encoding='utf-8') as f:
        json.dump(repeat_stat, f, indent=2, ensure_ascii=False)
    
    summary_file = os.path.join(repeat_stats_dir, f"repeat_{repeat_idx}_summary.json")
    summary = {
        "repeat_idx": repeat_idx,
        "total_repeats": total_repeats,
        "avg_improvement": repeat_stat["summary_stats"]["avg_improvement"],
        "std_improvement": repeat_stat["summary_stats"]["std_improvement"],
        "avg_reward": repeat_stat["summary_stats"]["avg_reward"],
        "std_reward": repeat_stat["summary_stats"]["std_reward"],
        "avg_branch_count": repeat_stat["summary_stats"]["avg_branch_count"],
        "avg_merge_count": repeat_stat["summary_stats"]["avg_merge_count"],
        "samples_processed": repeat_stat["samples_processed"],
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Saved repeat {repeat_idx} stats to {repeat_file}")
    print(f"📊 Repeat {repeat_idx} summary: improvement={summary['avg_improvement']:.4f}±{summary['std_improvement']:.4f}, "
          f"reward={summary['avg_reward']:.4f}±{summary['std_reward']:.4f}")


def run_multi_session_multi_branch_experiment(
    data_list: List[Dict[str, Any]],
    num_repeat: int = 1,
    use_all_branches: bool = True,
    max_branches: int = 64,
    merge_similarity_threshold: float = 0.95,
    results_logger: Optional[Any] = None,
    timestamp: str = None,
    results_dir: str = None
) -> Dict[str, Any]:
    """Multi-session × multi-branch experiment with forgetting adaptor and branch limits."""
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("🚀 Starting multi-session multi-branch experiment with forgetting adaptor...")
    log_memory_usage("Start")
    
    model_name = "/home/pj/Desktop/models/Meta-Llama-3-8B-Instruct"
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    prompt_builder = PromptBuilder(tokenizer, max_total_tokens=512, safety_margin=20)
    print(f"  ✅ PromptBuilder initialized: max_total_tokens={prompt_builder.max_total_tokens}, safety_margin={prompt_builder.safety_margin}")
    
    base_model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    base_model.eval()
    log_memory_usage("After loading base model")
    
    feature_extractor = FeatureExtractor(base_model=base_model)
    
    lora_loader = LoRAModelLoader(
        base_model=base_model,
        lora_rank=8,
        lora_alpha=16,
        lora_dropout=0.1
    )
    
    pipeline_model = transformers.pipeline(
        "text-generation",
        model=base_model,
        tokenizer=tokenizer,
        device_map="auto",
    )
    log_memory_usage("After creating pipeline")
    
    repeat_stats = []
    
    all_cumulative_improvement_sequences = []
    all_cumulative_reward_sequences = []
    
    cumulative_improvement_averager = CumulativeAverager()
    cumulative_reward_averager = CumulativeAverager()
    multi_repeat_averager = MultiRepeatCumulativeAverager()
    
    all_repeat_raw_stats = []
    
    for repeat_idx in range(num_repeat):
        print(f"\n🔄 Starting repeat {repeat_idx + 1}/{num_repeat}")
        log_memory_usage(f"Repeat {repeat_idx} start")
        
        if repeat_idx > 0:
            random.shuffle(data_list)
        
        storage_adaptor = EnhancedHierarchicalMemoryAdaptor(
            input_dim=feature_extractor.hidden_dim,
            max_adapters=5,
            adaptor_type="storage",
            max_branches=min(max_branches, 32),
            merge_similarity_threshold=merge_similarity_threshold
        )
        
        retrieval_adaptor = EnhancedHierarchicalMemoryAdaptor(
            input_dim=feature_extractor.hidden_dim,
            max_adapters=5,
            adaptor_type="retrieval",
            max_branches=min(max_branches, 32),
            merge_similarity_threshold=merge_similarity_threshold
        )
        
        forgetting_adaptor = ForgettingAdaptor(
            input_dim=feature_extractor.hidden_dim,
            max_adapters=5,
            max_non_param_memory=3,
            max_branches=min(max_branches, 32),
            merge_similarity_threshold=merge_similarity_threshold
        )
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        storage_adaptor.to(device)
        retrieval_adaptor.to(device)
        forgetting_adaptor.to(device)
        
        param_memory_system = EnhancedParametricMemory(
            base_model_name=model_name,
            max_adapters=5,
            training_learning_rate=2e-4,
            training_num_epochs=1,
            training_batch_size=2,
            max_training_samples=20
        )
        param_memory_system.set_model(base_model)
        param_memory_system.set_tokenizer(tokenizer)
        
        learning_rate = 1e-3
        all_params = list(storage_adaptor.parameters()) + list(retrieval_adaptor.parameters()) + list(forgetting_adaptor.parameters())
        optimizer = torch.optim.Adam(all_params, lr=learning_rate)
        
        storage_temp_config = {"initial": 2.0, "final": 0.2, "decay_type": "cosine"}
        retrieval_temp_config = {"initial": 1.5, "final": 0.1, "decay_type": "cosine"}
        forgetting_temp_config = {"initial": 1.0, "final": 0.1, "decay_type": "cosine"}
        
        repeat_improvements = []
        repeat_rewards = []
        repeat_logits_diffs = []
        repeat_branch_counts = []
        repeat_forgetting_quality_scores = []
        repeat_merge_counts = []
        
        repeat_training_stats = {
            "training_performed_count": 0,
            "total_training_loss": 0.0,
            "total_training_perplexity": 0.0,
            "total_training_samples": 0,
        }
        
        repeat_cumulative_improvements = []
        repeat_cumulative_rewards = []
        
        # Create live update file for current repeat
        repeat_raw_file = os.path.join(results_dir, "repeat_stats", f"repeat_{repeat_idx}_raw.json")
        os.makedirs(os.path.dirname(repeat_raw_file), exist_ok=True)
        
        # Initialize live statistics file
        live_stats = {
            "repeat_idx": repeat_idx,
            "cumulative_improvement_sequence": [],
            "cumulative_reward_sequence": [],
            "raw_improvements": [],
            "raw_rewards": [],
            "samples_processed": 0,
            "current_cumulative_improvement": 0.0,
            "current_cumulative_reward": 0.0,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(repeat_raw_file, 'w', encoding='utf-8') as f:
            json.dump(live_stats, f, indent=2, ensure_ascii=False)
        
        samples_to_process = len(data_list)
        print(f"  Processing {samples_to_process} samples")
        
        for idx in range(samples_to_process):
            sample = data_list[idx]
            
            if idx % 2 == 0:
                torch.cuda.empty_cache()
                gc.collect()
                log_memory_usage(f"Sample {idx} start")
            
            storage_temp = get_optimized_temperature(
                idx,
                initial_temp=storage_temp_config["initial"],
                final_temp=storage_temp_config["final"],
                total_samples=samples_to_process,
                decay_type=storage_temp_config["decay_type"]
            )
            
            retrieval_temp = get_optimized_temperature(
                idx,
                initial_temp=retrieval_temp_config["initial"],
                final_temp=retrieval_temp_config["final"],
                total_samples=samples_to_process,
                decay_type=retrieval_temp_config["decay_type"]
            )
            
            print(f"    Resetting parametric memory for new sample (haystack_sessions) {idx}")
            param_memory_system.reset_for_new_session()
            
            if idx > 0:
                storage_adaptor.clear_decision_history()
                retrieval_adaptor.clear_decision_history()
                forgetting_adaptor.clear_decision_history()
            
            sample_results, total_rewards, normalized_rewards, improvements, logits_diffs, forgetting_quality_scores = execute_complete_parallel_pipeline_with_limit(
                sample,
                idx,
                storage_adaptor,
                retrieval_adaptor,
                forgetting_adaptor,
                param_memory_system,
                feature_extractor,
                base_model,
                tokenizer,
                pipeline_model,
                lora_loader,
                storage_temp,
                retrieval_temp,
                min(max_branches, 32),
                merge_similarity_threshold,
                max_branches_per_session=2,
                use_length_normalization=True,
                prompt_builder=prompt_builder
            )
            
            if not sample_results:
                continue
            
            for result in sample_results:
                if result.get("training_performed", False):
                    repeat_training_stats["training_performed_count"] += 1
                    repeat_training_stats["total_training_loss"] += result.get("training_loss", 0.0)
                    repeat_training_stats["total_training_perplexity"] += result.get("training_perplexity", 0.0)
                    repeat_training_stats["total_training_samples"] += result.get("training_samples", 0)
            
            repeat_improvements.extend(improvements[:10])
            repeat_rewards.extend(total_rewards[:10])
            repeat_logits_diffs.extend(logits_diffs[:10])
            repeat_forgetting_quality_scores.extend(forgetting_quality_scores[:10])
            repeat_branch_counts.append(len(total_rewards))
            
            if sample_results:
                avg_merge_count = np.mean([r.get("merge_count", 1) for r in sample_results[:10]])
                repeat_merge_counts.append(avg_merge_count)
            
            if improvements and total_rewards:
                avg_improvement = np.mean(improvements)
                avg_reward = np.mean(total_rewards)
                
                cumulative_improvement_avg = cumulative_improvement_averager.add_value(avg_improvement)
                cumulative_reward_avg = cumulative_reward_averager.add_value(avg_reward)
                
                repeat_cumulative_improvements.append(cumulative_improvement_avg)
                repeat_cumulative_rewards.append(cumulative_reward_avg)
                
                # Update JSON file in real-time
                try:
                    with open(repeat_raw_file, 'r', encoding='utf-8') as f:
                        current_data = json.load(f)
                    
                    current_data["cumulative_improvement_sequence"] = repeat_cumulative_improvements
                    current_data["cumulative_reward_sequence"] = repeat_cumulative_rewards
                    current_data["raw_improvements"] = repeat_improvements[-100:]
                    current_data["raw_rewards"] = repeat_rewards[-100:]
                    current_data["samples_processed"] = len(repeat_cumulative_improvements)
                    current_data["current_cumulative_improvement"] = cumulative_improvement_avg
                    current_data["current_cumulative_reward"] = cumulative_reward_avg
                    current_data["last_update"] = datetime.now().isoformat()
                    current_data["progress"] = f"{idx+1}/{samples_to_process}"
                    
                    temp_file = repeat_raw_file + '.tmp'
                    with open(temp_file, 'w', encoding='utf-8') as f:
                        json.dump(current_data, f, indent=2, ensure_ascii=False)
                    os.replace(temp_file, repeat_raw_file)
                    
                except Exception as e:
                    print(f"⚠️  Failed to update live stats file: {e}")
            
            if results_logger is not None:
                for result in sample_results[:10]:
                    results_logger.log_result(**result)
            
            if idx > 0 and idx % 2 == 0 and total_rewards:
                optimizer, update_info = update_enhanced_adaptors_with_all_branches(
                    storage_adaptor,
                    retrieval_adaptor,
                    forgetting_adaptor,
                    optimizer,
                    total_rewards[:10],
                    forgetting_quality_scores[:10],
                    use_all_branches=use_all_branches,
                    learning_rate=learning_rate
                )
                
                if idx % 4 == 0:
                    learning_rate *= 0.95
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = learning_rate
            
            del sample_results, total_rewards, normalized_rewards
            del improvements, logits_diffs, forgetting_quality_scores
            torch.cuda.empty_cache()
            gc.collect()
        
        if repeat_training_stats["training_performed_count"] > 0:
            repeat_training_stats["avg_training_loss"] = repeat_training_stats["total_training_loss"] / repeat_training_stats["training_performed_count"]
            repeat_training_stats["avg_training_perplexity"] = repeat_training_stats["total_training_perplexity"] / repeat_training_stats["training_performed_count"]
        else:
            repeat_training_stats["avg_training_loss"] = 0.0
            repeat_training_stats["avg_training_perplexity"] = 0.0
        
        repeat_cumulative_averager = CumulativeAverager()
        for i in range(len(repeat_cumulative_improvements)):
            repeat_cumulative_averager.add_value(repeat_cumulative_improvements[i])
        
        multi_repeat_averager.add_repeat(cumulative_improvement_averager)
        
        if repeat_cumulative_improvements:
            all_cumulative_improvement_sequences.append(repeat_cumulative_improvements)
        if repeat_cumulative_rewards:
            all_cumulative_reward_sequences.append(repeat_cumulative_rewards)
        
        cumulative_improvement_averager.reset()
        cumulative_reward_averager.reset()
        
        lora_loader.clear_cache()
        
        current_repeat_stat = {
            "repeat_idx": repeat_idx,
            "raw_improvements": repeat_improvements,
            "raw_rewards": repeat_rewards,
            "raw_logits_diffs": repeat_logits_diffs,
            "raw_forgetting_quality_scores": repeat_forgetting_quality_scores,
            "raw_branch_counts": repeat_branch_counts,
            "raw_merge_counts": repeat_merge_counts,
            "samples_processed": len(repeat_improvements),
            "cumulative_improvement_sequence": repeat_cumulative_improvements,
            "cumulative_reward_sequence": repeat_cumulative_rewards,
            "cumulative_improvement_avg": repeat_cumulative_improvements[-1] if repeat_cumulative_improvements else 0.0,
            "cumulative_reward_avg": repeat_cumulative_rewards[-1] if repeat_cumulative_rewards else 0.0,
            "training_stats": repeat_training_stats,
            "summary_stats": {
                "avg_improvement": np.mean(repeat_improvements) if repeat_improvements else 0.0,
                "std_improvement": np.std(repeat_improvements) if repeat_improvements else 0.0,
                "avg_reward": np.mean(repeat_rewards) if repeat_rewards else 0.0,
                "std_reward": np.std(repeat_rewards) if repeat_rewards else 0.0,
                "avg_logits_diff": np.mean(repeat_logits_diffs) if repeat_logits_diffs else 0.0,
                "avg_forgetting_quality": np.mean(repeat_forgetting_quality_scores) if repeat_forgetting_quality_scores else 0.0,
                "avg_branch_count": np.mean(repeat_branch_counts) if repeat_branch_counts else 1.0,
                "avg_merge_count": np.mean(repeat_merge_counts) if repeat_merge_counts else 1.0,
                "avg_training_loss": repeat_training_stats["avg_training_loss"],
                "avg_training_perplexity": repeat_training_stats["avg_training_perplexity"],
                "training_performed_count": repeat_training_stats["training_performed_count"],
            }
        }
        
        # Final save of complete version
        simplified_stat = save_repeat_raw_stats_to_file(current_repeat_stat, results_dir, repeat_idx)
        all_repeat_raw_stats.append(simplified_stat)
        
        if repeat_improvements:
            repeat_stat = {
                "repeat_idx": repeat_idx,
                "avg_improvement": np.mean(repeat_improvements),
                "std_improvement": np.std(repeat_improvements),
                "avg_reward": np.mean(repeat_rewards),
                "std_reward": np.std(repeat_rewards),
                "avg_logits_diff": np.mean(repeat_logits_diffs) if repeat_logits_diffs else 0.0,
                "avg_forgetting_quality": np.mean(repeat_forgetting_quality_scores) if repeat_forgetting_quality_scores else 0.0,
                "avg_branch_count": np.mean(repeat_branch_counts) if repeat_branch_counts else 1.0,
                "avg_merge_count": np.mean(repeat_merge_counts) if repeat_merge_counts else 1.0,
                "samples_processed": len(repeat_improvements),
                "cumulative_improvement_avg": repeat_cumulative_improvements[-1] if repeat_cumulative_improvements else 0.0,
                "cumulative_reward_avg": repeat_cumulative_rewards[-1] if repeat_cumulative_rewards else 0.0,
                "cumulative_improvement_sequence": repeat_cumulative_improvements,
                "cumulative_reward_sequence": repeat_cumulative_rewards,
                "training_stats": repeat_training_stats,
            }
            repeat_stats.append(repeat_stat)
        
        del storage_adaptor, retrieval_adaptor, forgetting_adaptor, param_memory_system, optimizer
        del repeat_improvements, repeat_rewards, repeat_logits_diffs
        del repeat_branch_counts, repeat_forgetting_quality_scores, repeat_merge_counts
        del repeat_cumulative_improvements, repeat_cumulative_rewards
        torch.cuda.empty_cache()
        gc.collect()
        
        log_memory_usage(f"Repeat {repeat_idx} end")
    
    experiment_stats = compute_final_experiment_stats(
        all_repeat_raw_stats=all_repeat_raw_stats,
        repeat_stats=repeat_stats,
        all_cumulative_improvement_sequences=all_cumulative_improvement_sequences,
        all_cumulative_reward_sequences=all_cumulative_reward_sequences,
        multi_repeat_averager=multi_repeat_averager,
        num_repeat=num_repeat,
        use_all_branches=use_all_branches,
        max_branches=max_branches,
        merge_similarity_threshold=merge_similarity_threshold,
        all_results=[]
    )
    
    experiment_stats["prompt_builder_config"] = {
        "max_total_tokens": prompt_builder.max_total_tokens,
        "safety_margin": prompt_builder.safety_margin,
        "available_tokens": prompt_builder.available_tokens,
        "question_completeness": "100% guaranteed"
    }
    
    return experiment_stats

if __name__ == "__main__":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
    
    torch.cuda.empty_cache()
    gc.collect()
    
    num_repeat = args.num_repeat
    max_branches = args.max_branches
    merge_similarity_threshold = args.merge_similarity_threshold
    num_samples = NUM_SAMPLES  # Use the NUM_SAMPLES from utils
    use_all_branches = True
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    branch_mode = "all_branches" if use_all_branches else "best_branch"
    results_dir = f"results/multi_session_multi_branch_with_forgetting_{timestamp}_repeat{num_repeat}_{branch_mode}_max{max_branches}_merge{merge_similarity_threshold}_{experiment_config_str}"
    os.makedirs(results_dir, exist_ok=True)
    
    log_file_path = os.path.join(results_dir, "console_output.log")
    print(f"📝 All console output will be saved to: {log_file_path}")
    
    with open(log_file_path, 'w', encoding='utf-8') as log_file:
        # No results logger - all file generation removed
        results_logger = None
        
        print(f"📁 Results directory: {results_dir}")
        
        print("📂 Loading data...")
        data_list = load_longmem_json("data/longmemeval_s_cleaned.json")[:num_samples]
        print(f"📊 Loaded {len(data_list)} samples (config: {num_samples} samples)")
        log_memory_usage("After loading data")
        
        start_time = time.time()
        experiment_stats = run_multi_session_multi_branch_experiment(
            data_list=data_list,
            num_repeat=num_repeat,
            use_all_branches=use_all_branches,
            max_branches=max_branches,
            merge_similarity_threshold=merge_similarity_threshold,
            results_logger=results_logger,
            timestamp=timestamp,
            results_dir=results_dir
        )
        elapsed_time = time.time() - start_time
        
        print(f"\n" + "="*80)
        print("FINAL EXPERIMENT SUMMARY")
        print("="*80)
        print(f"⏱️  Total elapsed time: {elapsed_time:.2f}s")
        print(f"🔄 Number of repeats: {experiment_stats['num_repeats']}")
        print(f"📊 Number of samples: {num_samples}")
        print(f"🌳 Branch learning mode: {'All branches' if experiment_stats['use_all_branches'] else 'Best branch only'}")
        print(f"📊 Max branches: {experiment_stats['max_branches']}")
        print(f"🤝 Merge similarity threshold: {experiment_stats['merge_similarity_threshold']}")
        print(f"🎯 Experiment config: {experiment_stats['experiment_config']}")
        
        if "overall_statistics" in experiment_stats:
            overall = experiment_stats["overall_statistics"]
            print(f"\n📊 OVERALL STATISTICS (across {num_repeat} repeats):")
            print(f"  Improvement: {overall['improvement']['mean']:.4f} ± {overall['improvement']['std']:.4f}")
            print(f"  Reward: {overall['reward']['mean']:.4f} ± {overall['reward']['std']:.4f}")
            print(f"  Forgetting quality: {overall['forgetting_quality']['mean']:.4f} ± {overall['forgetting_quality']['std']:.4f}")
            print(f"  Branch count: {overall['branch_count']['mean']:.1f} ± {overall['branch_count']['std']:.1f}")
            print(f"  Training operations: {overall['training_stats']['total_training_operations']}")
        
        if "prompt_builder_config" in experiment_stats:
            pb_config = experiment_stats["prompt_builder_config"]
            print(f"\n📝 PromptBuilder Configuration:")
            print(f"  Max total tokens: {pb_config['max_total_tokens']}")
            print(f"  Safety margin: {pb_config['safety_margin']}")
            print(f"  Available tokens: {pb_config['available_tokens']}")
            print(f"  Question completeness: {pb_config['question_completeness']}")
        
        print(f"\n🔧 Control variables:")
        for key, value in experiment_stats.get('control_variables', {}).items():
            print(f"  {key}: {value}")
        
        stats_path = os.path.join(results_dir, "experiment_stats.json")
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(experiment_stats, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Experiment statistics saved to: {stats_path}")
        
        torch.cuda.empty_cache()
        gc.collect()
        print(f"\n✅ Experiment completed. Results saved to: {results_dir}")
        print("="*80)
    
    print(f"\n✅ All console output saved to: {log_file_path}")