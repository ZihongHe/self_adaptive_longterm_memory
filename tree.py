"""
Multi-branch decision tree structures and management utilities.

This module defines the data structures for representing multi-branch
decision paths, including nodes, branch manager, and the main decision
tree class. It also provides helper functions for aligning decisions
with nodes and creating default decision records.
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass
import time
import hashlib
import uuid
import torch
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
import numpy as np
import math
import uuid


# Multi-branch decision tree data structure
@dataclass
class MultiBranchDecisionTreeNode:
    """Multi-branch decision tree node for storing decision paths and states"""
    node_id: str
    features: torch.Tensor
    decision_type: str
    decision_result: Any
    log_prob: torch.Tensor
    temperature: float
    stable_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    reward: float = 0.0
    cumulative_reward: float = 0.0
    normalized_reward: float = 0.0
    branch_average_reward: float = 0.0
    children: List['MultiBranchDecisionTreeNode'] = field(default_factory=list)
    parent: Optional['MultiBranchDecisionTreeNode'] = None
    is_best_branch: bool = False
    path_id: str = ""
    branch_idx: int = 0
    session_idx: int = 0
    is_leaf: bool = False
    forward_output: Any = None
    inference_result: Any = None
    reward_calculated: bool = False
    memory_state_before: Optional[Dict[str, Any]] = None
    memory_state_after: Optional[Dict[str, Any]] = None
    parametric_memory_state: Optional[Any] = None
    non_parametric_memory_state: Optional[List[str]] = None
    
    merged_from: List[str] = field(default_factory=list)
    merge_count: int = 1
    decision_hash: str = ""
    
    token_info: Optional[Dict[str, Any]] = None
    length_adjusted: bool = False
    
    def __post_init__(self):
        if not hasattr(self, 'merge_count') or self.merge_count < 1:
            self.merge_count = 1
        if self.merged_from is None:
            self.merged_from = []
        if self.children is None:
            self.children = []
        if self.token_info is None:
            self.token_info = {}
    
    def __eq__(self, other):
        if not isinstance(other, MultiBranchDecisionTreeNode):
            return False
        return self.stable_id == other.stable_id
    
    def __hash__(self):
        return hash(self.stable_id)
    
    def add_child(self, child: 'MultiBranchDecisionTreeNode'):
        child.parent = self
        self.children.append(child)
    
    def get_path(self) -> List['MultiBranchDecisionTreeNode']:
        path = []
        node = self
        while node is not None:
            path.append(node)
            node = node.parent
        return list(reversed(path))
    
    def get_branch_log_probs(self) -> List[torch.Tensor]:
        path = self.get_path()
        return [node.log_prob for node in path if node.log_prob is not None]
    
    def update_cumulative_reward(self, reward: float, normalized_reward: float = 0.0):
        self.reward = reward
        self.normalized_reward = normalized_reward
        self.reward_calculated = True
        self.cumulative_reward = reward
    
    def update_token_info(self, baseline_token_count: int, memory_token_count: int):
        self.token_info = {
            "baseline_token_count": baseline_token_count,
            "memory_token_count": memory_token_count,
            "token_count_ratio": memory_token_count / max(baseline_token_count, 1),
            "updated_time": time.time() if 'time' in globals() else 0,
        }
        self.length_adjusted = True
    
    def get_token_info(self) -> Dict[str, Any]:
        if self.token_info is None:
            return {}
        return self.token_info.copy()
    
    def mark_as_leaf(self):
        self.is_leaf = True
    
    def count_nodes(self) -> int:
        count = 1
        for child in self.children:
            count += child.count_nodes()
        return count
    
    def mark_best_branch(self, is_best: bool = True):
        self.is_best_branch = is_best
        for child in self.children:
            child.mark_best_branch(is_best)
    
    def get_memory_state(self) -> Dict[str, Any]:
        return {
            "parametric": self.parametric_memory_state,
            "non_parametric": self.non_parametric_memory_state,
            "path_id": self.path_id,
            "branch_idx": self.branch_idx,
            "session_idx": self.session_idx,
            "merge_count": self.merge_count,
            "merged_from": self.merged_from,
            "token_info": self.token_info,
            "length_adjusted": self.length_adjusted,
            "stable_id": self.stable_id,
        }
    
    def get_decision_similarity(self, other: 'MultiBranchDecisionTreeNode') -> float:
        if self == other:
            return 1.0
        
        similarity = 0.0
        
        if self.decision_type == other.decision_type:
            similarity += 0.2
        
        try:
            if str(self.decision_result) == str(other.decision_result):
                similarity += 0.2
            elif isinstance(self.decision_result, (int, float)) and isinstance(other.decision_result, (int, float)):
                if abs(self.decision_result - other.decision_result) < 0.1:
                    similarity += 0.15
        except:
            pass
        
        temp_diff = abs(self.temperature - other.temperature)
        if temp_diff < 0.1:
            similarity += 0.1
        elif temp_diff < 0.5:
            similarity += 0.05
        
        if self.path_id == other.path_id:
            similarity += 0.1
        elif self.path_id and other.path_id and self.path_id.split('_')[0] == other.path_id.split('_')[0]:
            similarity += 0.05
        
        if self.features is not None and other.features is not None:
            try:
                cos_sim = F.cosine_similarity(
                    self.features.view(1, -1).float(),
                    other.features.view(1, -1).float(),
                    dim=1
                ).item()
                
                if math.isnan(cos_sim):
                    cos_sim = 0.0
                
                normalized_cos_sim = (cos_sim + 1) / 2
                similarity += normalized_cos_sim * 0.4
                
            except Exception:
                similarity += 0.2
        elif self.features is None and other.features is None:
            similarity += 0.2
        
        return max(0.0, min(1.0, similarity))

class BranchManager:
    """Branch manager maintaining mapping from stable_id to nodes, ensuring branch identity consistency."""
    
    def __init__(self):
        self.stable_id_to_node: Dict[str, MultiBranchDecisionTreeNode] = {}
        self.stable_id_to_reward: Dict[str, float] = {}
        self.stable_id_to_normalized_reward: Dict[str, float] = {}
        self.active_stable_ids: List[str] = []
    
    def register_node(self, node: MultiBranchDecisionTreeNode):
        """Register node to mapping"""
        self.stable_id_to_node[node.stable_id] = node
        if node.stable_id not in self.active_stable_ids:
            self.active_stable_ids.append(node.stable_id)
    
    def unregister_node(self, stable_id: str):
        """Unregister node when branch is merged or deleted"""
        if stable_id in self.active_stable_ids:
            self.active_stable_ids.remove(stable_id)
    
    def get_node_by_stable_id(self, stable_id: str) -> Optional[MultiBranchDecisionTreeNode]:
        """Get node by stable_id"""
        return self.stable_id_to_node.get(stable_id)
    
    def get_active_nodes(self) -> List[MultiBranchDecisionTreeNode]:
        """Get all active nodes"""
        return [self.stable_id_to_node[sid] for sid in self.active_stable_ids if sid in self.stable_id_to_node]
    
    def set_reward(self, stable_id: str, reward: float, normalized_reward: float):
        """Set reward for branch"""
        self.stable_id_to_reward[stable_id] = reward
        self.stable_id_to_normalized_reward[stable_id] = normalized_reward
        
        node = self.get_node_by_stable_id(stable_id)
        if node:
            node.update_cumulative_reward(reward, normalized_reward)
    
    def get_rewards(self, stable_ids: List[str] = None) -> Tuple[List[float], List[float], List[str]]:
        """Get rewards for specified branches"""
        if stable_ids is None:
            stable_ids = self.active_stable_ids
        
        rewards = []
        normalized_rewards = []
        valid_stable_ids = []
        
        for sid in stable_ids:
            if sid in self.stable_id_to_reward:
                rewards.append(self.stable_id_to_reward[sid])
                normalized_rewards.append(self.stable_id_to_normalized_reward[sid])
                valid_stable_ids.append(sid)
            else:
                rewards.append(0.0)
                normalized_rewards.append(0.0)
                valid_stable_ids.append(sid)
        
        return rewards, normalized_rewards, valid_stable_ids
    
    def get_aligned_branches_and_rewards(self, nodes: List[MultiBranchDecisionTreeNode] = None) -> Tuple[List[MultiBranchDecisionTreeNode], List[float], List[float]]:
        """Get aligned branches and rewards"""
        if nodes is None:
            nodes = self.get_active_nodes()
        
        aligned_nodes = []
        aligned_rewards = []
        aligned_normalized_rewards = []
        
        for node in nodes:
            if node.stable_id in self.stable_id_to_reward:
                aligned_nodes.append(node)
                aligned_rewards.append(self.stable_id_to_reward[node.stable_id])
                aligned_normalized_rewards.append(self.stable_id_to_normalized_reward[node.stable_id])
            else:
                aligned_nodes.append(node)
                aligned_rewards.append(0.0)
                aligned_normalized_rewards.append(0.0)
        
        return aligned_nodes, aligned_rewards, aligned_normalized_rewards
    
    def sync_with_tree(self, tree):
        """Synchronize active nodes with decision tree"""
        self.active_stable_ids = []
        for node in tree.current_nodes:
            self.register_node(node)
            self._register_subtree_nodes(node)
    
    def _register_subtree_nodes(self, node: MultiBranchDecisionTreeNode):
        """Recursively register all nodes in subtree"""
        self.stable_id_to_node[node.stable_id] = node
        for child in node.children:
            self._register_subtree_nodes(child)
    
    def clear(self):
        """Clear all records"""
        self.stable_id_to_node.clear()
        self.stable_id_to_reward.clear()
        self.stable_id_to_normalized_reward.clear()
        self.active_stable_ids.clear()



class MultiBranchDecisionTree:
    """Multi-branch decision tree manager supporting branch limits and merging, using stable_id for branch identity."""
    
    def __init__(self, max_branches: int = 64, merge_similarity_threshold: float = 0.8):
        self.root: Optional[MultiBranchDecisionTreeNode] = None
        self.current_nodes: List[MultiBranchDecisionTreeNode] = []
        self.leaf_nodes: List[MultiBranchDecisionTreeNode] = []
        self.decision_sequences: List[List[MultiBranchDecisionTreeNode]] = []
        self.best_branch_path: Optional[List[MultiBranchDecisionTreeNode]] = None
        self.all_branch_rewards: List[float] = []
        self.all_branch_nodes: List[List[MultiBranchDecisionTreeNode]] = []
        self.path_counter: int = 0
        self.session_counter: int = 0
        self.path_id_map: Dict[str, MultiBranchDecisionTreeNode] = {}
        self.branch_memory_states: Dict[str, Dict[str, Any]] = {}
        self.branch_counter: int = 0
        self.max_branches: int = max_branches
        self.merge_similarity_threshold: float = merge_similarity_threshold
        self.branch_usage_count: Dict[str, int] = {}
        
        self.branch_manager = BranchManager()
        
    def get_total_branches(self) -> int:
        """Get current total number of branches considering merges."""
        total = 0
        for node in self.current_nodes:
            total += node.merge_count
        return total
    
    def can_create_more_branches(self, num_new_branches: int) -> bool:
        """Check if more branches can be created"""
        current_total = self.get_total_branches()
        return current_total + num_new_branches <= self.max_branches
    
    def add_root(self, node: MultiBranchDecisionTreeNode):
        """Add root node"""
        self.root = node
        self.current_nodes = [node]
        self._register_path_id(node)
        self.branch_manager.register_node(node)
    
    def _register_path_id(self, node: MultiBranchDecisionTreeNode):
        """Register path ID"""
        if node.path_id:
            self.path_id_map[node.path_id] = node

    def expand_all_current_with_map(self, 
                                child_nodes_list: List[List[MultiBranchDecisionTreeNode]]) -> List[MultiBranchDecisionTreeNode]:
        """MAP expansion for all current nodes using stable_id for mapping."""
        if not self.current_nodes:
            print("⚠️  Warning: No current nodes to expand")
            return []
        
        new_current_nodes = []
        total_children_added = 0
        
        parent_to_children = {}
        
        for i in range(len(self.current_nodes)):
            parent_node = self.current_nodes[i]
            
            if i < len(child_nodes_list) and child_nodes_list[i]:
                child_nodes = child_nodes_list[i]
                children_added = 0
                
                parent_children = []
                for child in child_nodes:
                    current_total = self.get_total_branches()
                    if current_total >= self.max_branches:
                        break
                    
                    parent_node.add_child(child)
                    self._register_path_id(child)
                    
                    self.branch_manager.register_node(child)
                    
                    new_current_nodes.append(child)
                    parent_children.append(child)
                    children_added += 1
                
                total_children_added += children_added
                parent_to_children[parent_node.stable_id] = [c.stable_id for c in parent_children]
                
                if children_added > 0:
                    if parent_node.stable_id in self.branch_manager.active_stable_ids:
                        self.branch_manager.active_stable_ids.remove(parent_node.stable_id)
                    continue
            
            new_current_nodes.append(parent_node)
            if parent_node.stable_id not in self.branch_manager.active_stable_ids:
                self.branch_manager.active_stable_ids.append(parent_node.stable_id)
        
        self.current_nodes = new_current_nodes
        current_total = self.get_total_branches()
        
        print(f"  ↳ Expanded {total_children_added} children, total branches: {current_total}")
        
        return self.current_nodes

    def split_branch_for_session_with_limit(self, 
                                        session_idx: int, 
                                        total_sessions: int,
                                        max_branches_per_session: int = 2) -> List[MultiBranchDecisionTreeNode]:
        """Force multi-branch expansion using stable_id."""
        current_branch_count = self.get_total_branches()
        
        if not self.current_nodes:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            root_node = MultiBranchDecisionTreeNode(
                node_id=f"root_session_{session_idx}",
                features=torch.randn(1536, device=device).cpu(),
                decision_type="root",
                decision_result="start",
                log_prob=torch.tensor(0.0),
                temperature=1.0,
                path_id="root",
                branch_idx=0,
                session_idx=session_idx,
                parametric_memory_state=None,
                non_parametric_memory_state=[],
                decision_hash=self._compute_decision_hash("root", "start", torch.randn(1536)),
                merge_count=1
            )
            self.add_root(root_node)
            return self.current_nodes
        
        available_branches = self.max_branches - current_branch_count
        
        if available_branches < 1:
            return self.current_nodes
        
        merged_nodes = self._merge_similar_branches_before_expansion(self.current_nodes)
        
        for node in merged_nodes:
            if node.stable_id not in self.branch_manager.active_stable_ids:
                self.branch_manager.active_stable_ids.append(node.stable_id)
        self.current_nodes = merged_nodes
        
        current_nodes_count = len(self.current_nodes)
        branches_per_node = max(2, min(max_branches_per_session, available_branches // max(1, current_nodes_count)))
        
        child_nodes_list = []
        total_new_branches = 0
        
        for i, parent_node in enumerate(self.current_nodes):
            child_nodes = []
            
            for b in range(branches_per_node):
                if not self.can_create_more_branches(1):
                    break
                
                parametric_state = parent_node.parametric_memory_state
                non_parametric_state = parent_node.non_parametric_memory_state.copy() if parent_node.non_parametric_memory_state else []
                
                branch_node = MultiBranchDecisionTreeNode(
                    node_id=f"session_{session_idx}_node_{i}_branch_{b}",
                    features=parent_node.features.clone() if parent_node.features is not None else torch.randn(1536),
                    decision_type="session_branch_split",
                    decision_result=f"branch_{b}",
                    log_prob=torch.tensor(0.0),
                    temperature=parent_node.temperature,
                    path_id=f"{parent_node.path_id}_s{session_idx}_b{b}" if parent_node.path_id else f"p{self.path_counter}_s{session_idx}_b{b}",
                    branch_idx=b,
                    session_idx=session_idx,
                    parametric_memory_state=parametric_state,
                    non_parametric_memory_state=non_parametric_state,
                    decision_hash=self._compute_decision_hash("session_branch_split", f"branch_{b}", parent_node.features),
                    merge_count=parent_node.merge_count
                )
                child_nodes.append(branch_node)
                total_new_branches += 1
            
            child_nodes_list.append(child_nodes)
        
        if not child_nodes_list:
            return self.current_nodes
        
        new_current_nodes = self.expand_all_current_with_map(child_nodes_list)
        
        self.session_counter += 1
        self.branch_counter = self.get_total_branches()
        
        return new_current_nodes

    def _compute_decision_hash(self, decision_type: str, decision_result: Any, features: torch.Tensor) -> str:
        """Compute decision hash value"""
        decision_str = f"{decision_type}_{decision_result}"
        if features is not None:
            features_data = features.cpu().numpy() if features.is_cuda else features.numpy()
            features_str = "_".join([f"{x:.3f}" for x in features_data[:5]])
            decision_str += f"_{features_str}"
        return hashlib.md5(decision_str.encode()).hexdigest()
    
    def _merge_similar_branches_before_expansion(self, nodes: List[MultiBranchDecisionTreeNode]) -> List[MultiBranchDecisionTreeNode]:
        """Merge similar branch nodes before decision sampling"""
        if len(nodes) <= 1:
            return nodes
        
        merged = [False] * len(nodes)
        merged_nodes = []
        
        for i in range(len(nodes)):
            if merged[i]:
                continue
            
            current_node = nodes[i]
            merge_group = [current_node]
            
            for j in range(i + 1, len(nodes)):
                if merged[j]:
                    continue
                
                other_node = nodes[j]
                similarity = current_node.get_decision_similarity(other_node)
                
                if similarity >= self.merge_similarity_threshold:
                    merge_group.append(other_node)
                    merged[j] = True
            
            if len(merge_group) > 1:
                main_node = merge_group[0]
                merged_from = []
                total_merge_count = main_node.merge_count
                
                for node in merge_group[1:]:
                    merged_from.append(node.stable_id)
                    total_merge_count += node.merge_count
                    self.branch_manager.unregister_node(node.stable_id)
                
                main_node.merged_from.extend(merged_from)
                main_node.merge_count = total_merge_count
                
                for node in merge_group[1:]:
                    if node.non_parametric_memory_state:
                        for mem in node.non_parametric_memory_state:
                            if (main_node.non_parametric_memory_state is None or 
                                mem not in main_node.non_parametric_memory_state):
                                if main_node.non_parametric_memory_state is None:
                                    main_node.non_parametric_memory_state = []
                                main_node.non_parametric_memory_state.append(mem)
                
                merged_nodes.append(main_node)
                merged[i] = True
            else:
                merged_nodes.append(current_node)
                merged[i] = True
        
        return merged_nodes
    
    def add_decision_nodes_with_map(self, 
                                  decision_type: str,
                                  decision_results: List[Any],
                                  log_probs: List[torch.Tensor],
                                  temperatures: List[float]) -> List[MultiBranchDecisionTreeNode]:
        """Add decision nodes using MAP approach"""
        if len(self.current_nodes) != len(decision_results):
            print(f"❌ Error: Mismatch in add_decision_nodes_with_map: {len(self.current_nodes)} nodes, {len(decision_results)} results. Discarding.")
            return []
        
        new_nodes_list = []
        
        for i, (current_node, decision_result, log_prob, temperature) in enumerate(zip(
            self.current_nodes, decision_results, log_probs, temperatures
        )):
            decision_node = MultiBranchDecisionTreeNode(
                node_id=f"{decision_type}_{time.time()}_{i}",
                features=current_node.features.clone() if current_node.features is not None else torch.randn(1536),
                decision_type=decision_type,
                decision_result=decision_result,
                log_prob=log_prob,
                temperature=temperature,
                path_id=current_node.path_id + f"_{decision_type}" if current_node.path_id else f"p{self.path_counter}_{decision_type}",
                branch_idx=current_node.branch_idx,
                session_idx=current_node.session_idx,
                parametric_memory_state=current_node.parametric_memory_state,
                non_parametric_memory_state=current_node.non_parametric_memory_state,
                merge_count=current_node.merge_count
            )
            new_nodes_list.append([decision_node])
        
        new_current_nodes = self.expand_all_current_with_map(new_nodes_list)
        
        return new_current_nodes
    
    def finalize_branch_with_reward(self, leaf_node: MultiBranchDecisionTreeNode, 
                                  reward: float, normalized_reward: float):
        """Finalize a branch and set its reward"""
        leaf_node.mark_as_leaf()
        leaf_node.update_cumulative_reward(reward, normalized_reward)
        
        if leaf_node not in self.leaf_nodes:
            self.leaf_nodes.append(leaf_node)
        
        self.branch_manager.set_reward(leaf_node.stable_id, reward, normalized_reward)
        
        path = leaf_node.get_path()
        path_ids = [node.node_id for node in path]
        
        path_exists = False
        for existing_path in self.decision_sequences:
            existing_path_ids = [node.node_id for node in existing_path]
            if existing_path_ids == path_ids:
                path_exists = True
                break
        
        if not path_exists:
            self.decision_sequences.append(path)
            self.all_branch_nodes.append(path)
        
        self.branch_memory_states[leaf_node.path_id] = leaf_node.get_memory_state()
    
    def finalize_all_branches_with_rewards(self, rewards: List[float], normalized_rewards: List[float]):
        """Finalize all branches with rewards using stable_id for alignment."""
        if not self.current_nodes:
            print("⚠️  Warning: No current nodes to finalize")
            return
        
        active_stable_ids = [node.stable_id for node in self.current_nodes]
        
        if len(rewards) != len(active_stable_ids):
            print(f"❌ Error: Mismatch in rewards ({len(rewards)}) and current nodes ({len(active_stable_ids)}). Cannot finalize.")
            
            if len(rewards) > len(active_stable_ids):
                print(f"  ↳ Truncating rewards from {len(rewards)} to {len(active_stable_ids)}")
                rewards = rewards[:len(active_stable_ids)]
                normalized_rewards = normalized_rewards[:len(active_stable_ids)]
            else:
                print(f"  ↳ Adding {len(active_stable_ids) - len(rewards)} default rewards")
                while len(rewards) < len(active_stable_ids):
                    rewards.append(0.0)
                    normalized_rewards.append(0.0)
        
        for i, node in enumerate(self.current_nodes):
            if i < len(rewards):
                self.finalize_branch_with_reward(node, rewards[i], normalized_rewards[i])
        
        print(f"  ↳ Finalized {len(self.current_nodes)} branches with rewards")
    
    def get_all_paths(self) -> List[List[MultiBranchDecisionTreeNode]]:
        """Get all complete paths"""
        return self.decision_sequences
    
    def select_best_branch(self) -> Optional[List[MultiBranchDecisionTreeNode]]:
        """Select branch with highest reward as best branch"""
        if not self.leaf_nodes:
            return None
        
        best_node = max(self.leaf_nodes, key=lambda node: node.cumulative_reward * node.merge_count)
        best_path = best_node.get_path()
        
        for node in best_path:
            node.is_best_branch = True
        
        self.best_branch_path = best_path
        return best_path
    
    def get_best_branch_log_probs(self) -> List[torch.Tensor]:
        """Get all log probabilities of best branch"""
        if self.best_branch_path is None:
            return []
        return [node.log_prob for node in self.best_branch_path if node.log_prob is not None]
    
    def get_all_branch_log_probs(self) -> List[List[torch.Tensor]]:
        """Get log probabilities of all branches"""
        all_log_probs = []
        for path in self.decision_sequences:
            log_probs = [node.log_prob for node in path if node.log_prob is not None]
            if log_probs:
                all_log_probs.append(log_probs)
        return all_log_probs
    
    def get_average_path_reward(self) -> float:
        """Get average reward of all paths"""
        if not self.leaf_nodes:
            return 0.0
        total_reward = sum(node.cumulative_reward * node.merge_count for node in self.leaf_nodes)
        total_branches = sum(node.merge_count for node in self.leaf_nodes)
        return total_reward / total_branches if total_branches > 0 else 0.0
    
    def get_branch_rewards(self) -> List[float]:
        """Get cumulative rewards of all branches"""
        rewards = [node.cumulative_reward for node in self.leaf_nodes]
        return rewards
    
    def update_branch_average_rewards(self, average_reward: float):
        """Update average reward reference for all branches"""
        for node in self.leaf_nodes:
            node.branch_average_reward = average_reward
    
    def get_path_count(self) -> int:
        """Get number of paths"""
        return len(self.leaf_nodes)
    
    def get_branch_count(self) -> int:
        """Get number of branches considering merges"""
        return self.get_total_branches()
    
    def execute_parallel_reward_calculation(self, reward_func, *args, **kwargs):
        """Strict MAP parallel execution: ensure alignment through branch manager."""
        rewards = []
        normalized_rewards = []
        
        if not self.current_nodes:
            return rewards, normalized_rewards
        
        active_nodes = self.current_nodes
        active_stable_ids = [node.stable_id for node in active_nodes]
        
        node_rewards = {}
        node_normalized_rewards = {}
        
        for node in active_nodes:
            try:
                reward, normalized_reward = reward_func(node, *args, **kwargs)
                node_rewards[node.stable_id] = reward * node.merge_count
                node_normalized_rewards[node.stable_id] = normalized_reward * node.merge_count
            except Exception as e:
                node_rewards[node.stable_id] = 0.0
                node_normalized_rewards[node.stable_id] = 0.0
        
        for sid in active_stable_ids:
            if sid in node_rewards:
                rewards.append(node_rewards[sid])
                normalized_rewards.append(node_normalized_rewards[sid])
            else:
                rewards.append(0.0)
                normalized_rewards.append(0.0)
        
        if len(rewards) != len(active_nodes):
            print(f"❌ Error: Parallel reward calculation mismatch: {len(rewards)} rewards, {len(active_nodes)} nodes. Discarding rewards.")
            return [], []
        
        for i, node in enumerate(active_nodes):
            if i < len(rewards):
                original_reward = rewards[i] / node.merge_count if node.merge_count > 0 else rewards[i]
                original_normalized_reward = normalized_rewards[i] / node.merge_count if node.merge_count > 0 else normalized_rewards[i]
                self.branch_manager.set_reward(node.stable_id, original_reward, original_normalized_reward)
        
        return rewards, normalized_rewards
    
    def visualize(self, max_depth: int = 3):
        """Visualize decision tree showing merge information and branch statistics."""
        if not self.root:
            return
        
        def print_node(node: MultiBranchDecisionTreeNode, depth: int, prefix: str):
            if depth > max_depth:
                return
                
            node_type = node.decision_type
            result_str = str(node.decision_result)[:30]
            reward_str = f" R:{node.reward:.3f}" if node.reward != 0 else ""
            cum_reward_str = f" CR:{node.cumulative_reward:.3f}" if node.cumulative_reward != 0 else ""
            norm_reward_str = f" NR:{node.normalized_reward:.3f}" if node.normalized_reward != 0 else ""
            best_str = " ★" if node.is_best_branch else ""
            path_str = f" [{node.path_id}]" if node.path_id else ""
            leaf_str = " 🍃" if node.is_leaf else ""
            merge_str = f" 🧬×{node.merge_count}" if node.merge_count > 1 else ""
            stable_str = f" [ID:{node.stable_id[:8]}]"
            
            print(f"{prefix}├─ [{node_type}]{best_str}{leaf_str}{merge_str}{stable_str} {result_str}{reward_str}{cum_reward_str}{norm_reward_str}{path_str}")
            
            child_prefix = prefix + "│  "
            for i, child in enumerate(node.children):
                if i == len(node.children) - 1:
                    print_node(child, depth + 1, prefix + "   ")
                else:
                    print_node(child, depth + 1, child_prefix)
        
        print("Decision tree structure:")
        print_node(self.root, 0, "")
    
    def sync_branch_manager(self):
        """Synchronize with branch manager"""
        self.branch_manager.sync_with_tree(self)


def create_default_storage_decision(node: MultiBranchDecisionTreeNode) -> Dict[str, Any]:
    """Create default storage decision"""
    return {
        "store": False,
        "use_existing": False,
        "create_new": False,
        "selected_adapter": None,
        "store_to_non_parametric": False,
        "non_parametric_reason": "default",
        "store_non_parametric_probability": 0.0,
        "store_non_parametric_log_prob": 0.0,
        "reason": "default_decision",
        "log_prob": 0.0,
        "store_probability": 0.0,
        "path_id": node.path_id,
        "branch_idx": node.branch_idx,
        "session_idx": node.session_idx,
        "merge_count": node.merge_count,
        "storage_to_non_parametric": False,
        "non_parametric_memory_usage": len(node.non_parametric_memory_state or []) / 20.0 if node.non_parametric_memory_state else 0.0,
        "training_performed": False,
        "training_loss": 0.0,
        "stable_id": node.stable_id,
    }


def create_default_retrieval_decision(node: MultiBranchDecisionTreeNode) -> Dict[str, Any]:
    """Create default retrieval decision"""
    return {
        "use_parametric": False,
        "selected_adapter": None,
        "adapter_index": -1,
        "reason": "default_retrieval",
        "log_prob": 0.0,
        "use_probability": 0.0,
        "nll_influence": 0.0,
        "path_id": node.path_id,
        "branch_idx": node.branch_idx,
        "session_idx": node.session_idx,
        "non_parametric_context": [],
        "non_param_relevance_scores": [],
        "non_param_decision_log_probs": [],
        "merge_count": node.merge_count,
        "retrieval_aligned": True,
        "stable_id": node.stable_id,
    }


def create_default_forgetting_decision(node: MultiBranchDecisionTreeNode) -> Dict[str, Any]:
    """Create default forgetting decision"""
    non_parametric_memory = node.non_parametric_memory_state or []
    return {
        "forget": False,
        "forget_type": None,
        "forget_non_param_idx": -1,
        "forget_param_idx": -1,
        "forget_param_name": None,
        "reason": "default",
        "log_prob": 0.0,
        "forget_probability": 0.0,
        "path_id": node.path_id,
        "branch_idx": node.branch_idx,
        "session_idx": node.session_idx,
        "non_param_memory_before": len(non_parametric_memory),
        "param_memory_before": 0,
        "merge_count": node.merge_count,
        "forgotten_non_param": "",
        "forgotten_param_adapter": None,
        "stable_id": node.stable_id,
    }



def smart_align_nodes_and_decisions(nodes: List[MultiBranchDecisionTreeNode], 
                                   decisions: List[Dict[str, Any]], 
                                   default_decision_creator,
                                   decision_type: str = "storage") -> Tuple[List[MultiBranchDecisionTreeNode], List[Dict[str, Any]]]:
    """Intelligent alignment of nodes and decisions using stable_id."""
    if not nodes:
        return [], []
    
    if not decisions:
        print(f"⚠️  Warning: No {decision_type} decisions to align with {len(nodes)} nodes. Creating default decisions.")
        default_decisions = [default_decision_creator(node) for node in nodes]
        return nodes, default_decisions
    
    if len(decisions) == len(nodes):
        return nodes, decisions
    
    print(f"⚠️  Warning: {decision_type} decisions mismatch: {len(decisions)} decisions, {len(nodes)} nodes.")
    
    node_dict = {node.stable_id: node for node in nodes}
    decision_dict = {}
    
    for decision in decisions:
        if "stable_id" in decision and decision["stable_id"] in node_dict:
            decision_dict[decision["stable_id"]] = decision
    
    if decision_dict:
        aligned_nodes = []
        aligned_decisions = []
        
        for node in nodes:
            if node.stable_id in decision_dict:
                aligned_nodes.append(node)
                aligned_decisions.append(decision_dict[node.stable_id])
            else:
                aligned_nodes.append(node)
                aligned_decisions.append(default_decision_creator(node))
        
        print(f"  ↳ Aligned using stable_id: {len(aligned_nodes)} nodes, {len(aligned_decisions)} decisions")
        return aligned_nodes, aligned_decisions
    
    if len(decisions) > len(nodes):
        print(f"  ↳ Truncating {len(decisions) - len(nodes)} excess {decision_type} decisions")
        truncated_decisions = decisions[:len(nodes)]
        
        for i, decision in enumerate(truncated_decisions):
            if i < len(nodes):
                decision["stable_id"] = nodes[i].stable_id
                decision[f"{decision_type}_aligned"] = True
        
        return nodes, truncated_decisions
    
    else:
        print(f"  ↳ Adding {len(nodes) - len(decisions)} default {decision_type} decisions")
        extended_decisions = decisions.copy()
        
        while len(extended_decisions) < len(nodes):
            node_idx = len(extended_decisions)
            if node_idx < len(nodes):
                node = nodes[node_idx]
                default_decision = default_decision_creator(node)
                default_decision["stable_id"] = node.stable_id
                default_decision[f"{decision_type}_aligned"] = True
                extended_decisions.append(default_decision)
        
        return nodes, extended_decisions


def align_all_decisions(
    storage_decisions: List[Dict[str, Any]],
    retrieval_decisions: List[Dict[str, Any]],
    forgetting_decisions: List[Dict[str, Any]],
    nodes: List[MultiBranchDecisionTreeNode]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[MultiBranchDecisionTreeNode]]:
    """Unified alignment of all decision types to nodes."""
    if not nodes:
        return [], [], [], []
    
    aligned_nodes, aligned_storage = smart_align_nodes_and_decisions(
        nodes, storage_decisions, create_default_storage_decision, "storage"
    )
    
    aligned_nodes, aligned_retrieval = smart_align_nodes_and_decisions(
        aligned_nodes, retrieval_decisions, create_default_retrieval_decision, "retrieval"
    )
    
    aligned_nodes, aligned_forgetting = smart_align_nodes_and_decisions(
        aligned_nodes, forgetting_decisions, create_default_forgetting_decision, "forgetting"
    )
    
    min_len = min(len(aligned_nodes), len(aligned_storage), len(aligned_retrieval), len(aligned_forgetting))
    
    if min_len < len(aligned_nodes):
        print(f"⚠️  Warning: Final alignment still has inconsistency. Truncating to {min_len} branches.")
        aligned_nodes = aligned_nodes[:min_len]
        aligned_storage = aligned_storage[:min_len]
        aligned_retrieval = aligned_retrieval[:min_len]
        aligned_forgetting = aligned_forgetting[:min_len]
    
    print(f"  ✅ Final alignment complete: {len(aligned_nodes)} nodes, "
          f"{len(aligned_storage)} storage decisions, "
          f"{len(aligned_retrieval)} retrieval decisions, "
          f"{len(aligned_forgetting)} forgetting decisions")
    
    return aligned_storage, aligned_retrieval, aligned_forgetting, aligned_nodes

