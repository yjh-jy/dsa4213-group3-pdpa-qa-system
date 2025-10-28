#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
dense_chunk_and_extract.py — Specialized chunking for dense retrieval training
- Creates training triples (query, positive, negative) from QA dataset
- Implements k-fold cross-validation splits by section
- Generates hard negatives using BM25 and dense retrieval
- Optimized for training sentence transformers
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Set
from collections import defaultdict
import numpy as np

class DenseTrainingDataGenerator:
    """Generate training data for dense retrieval fine-tuning."""
    
    def __init__(self, corpus_path: Path, qa_path: Path, output_dir: Path):
        """Initialize with corpus and QA data paths."""
        self.corpus_path = Path(corpus_path)
        self.qa_path = Path(qa_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        self.corpus_chunks = self._load_corpus()
        self.qa_data = self._load_qa_data()
        
        # Create mappings
        self.chunk_to_section = self._create_chunk_section_mapping()
        self.section_to_chunks = self._create_section_chunk_mapping()
        
    def _load_corpus(self) -> Dict[str, Dict]:
        """Load corpus chunks."""
        chunks = {}
        with self.corpus_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    chunk = json.loads(line)
                    chunks[chunk["chunk_id"]] = chunk
        return chunks
    
    def _load_qa_data(self) -> List[Dict]:
        """Load QA dataset."""
        qa_data = []
        with self.qa_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    qa_data.append(json.loads(line))
        return qa_data
    
    def _create_chunk_section_mapping(self) -> Dict[str, str]:
        """Map chunk IDs to their sections."""
        mapping = {}
        for chunk_id, chunk in self.corpus_chunks.items():
            section_key = f"Part{chunk.get('part', '')}_Section{chunk.get('section', '')}"
            mapping[chunk_id] = section_key
        return mapping
    
    def _create_section_chunk_mapping(self) -> Dict[str, List[str]]:
        """Map sections to their chunk IDs."""
        mapping = defaultdict(list)
        for chunk_id, section in self.chunk_to_section.items():
            mapping[section].append(chunk_id)
        return dict(mapping)
    
    def _extract_positive_chunks(self, qa_item: Dict) -> Set[str]:
        """Extract positive chunk IDs from QA item."""
        positive_chunks = set()
        for link in qa_item.get("corpus_links", []):
            if "chunk_id" in link:
                positive_chunks.add(link["chunk_id"])
        return positive_chunks
    
    def _generate_hard_negatives(self, query: str, positive_chunks: Set[str], 
                                num_negatives: int = 5) -> List[str]:
        """Generate hard negative chunks using BM25-style scoring."""
        # Simple hard negative generation based on text similarity
        query_words = set(query.lower().split())
        
        candidates = []
        for chunk_id, chunk in self.corpus_chunks.items():
            if chunk_id in positive_chunks:
                continue
                
            # Calculate simple overlap score
            chunk_words = set(chunk.get("text", "").lower().split())
            overlap = len(query_words.intersection(chunk_words))
            
            if overlap > 0:  # Only consider chunks with some overlap
                candidates.append((chunk_id, overlap))
        
        # Sort by overlap (descending) and take top candidates as hard negatives
        candidates.sort(key=lambda x: x[1], reverse=True)
        hard_negatives = [chunk_id for chunk_id, _ in candidates[:num_negatives * 2]]
        
        # Add some random negatives for diversity
        all_chunk_ids = list(self.corpus_chunks.keys())
        random_negatives = random.sample([cid for cid in all_chunk_ids 
                                        if cid not in positive_chunks and cid not in hard_negatives], 
                                       min(num_negatives, len(all_chunk_ids) - len(positive_chunks) - len(hard_negatives)))
        
        # Combine and shuffle
        negatives = hard_negatives[:num_negatives//2] + random_negatives[:num_negatives//2]
        random.shuffle(negatives)
        
        return negatives[:num_negatives]
    
    def create_training_triples(self, num_negatives_per_positive: int = 3) -> List[Dict]:
        """Create training triples (query, positive, negative)."""
        triples = []
        
        for qa_item in self.qa_data:
            query = qa_item["question_user"]
            positive_chunks = self._extract_positive_chunks(qa_item)
            
            if not positive_chunks:
                continue
            
            # Generate negatives
            negatives = self._generate_hard_negatives(query, positive_chunks, 
                                                    num_negatives_per_positive * len(positive_chunks))
            
            # Create triples for each positive
            for pos_chunk_id in positive_chunks:
                pos_text = self.corpus_chunks[pos_chunk_id].get("text", "")
                
                # Create multiple negatives per positive
                chunk_negatives = negatives[:num_negatives_per_positive]
                negatives = negatives[num_negatives_per_positive:]  # Remove used negatives
                
                for neg_chunk_id in chunk_negatives:
                    if neg_chunk_id in self.corpus_chunks:
                        neg_text = self.corpus_chunks[neg_chunk_id].get("text", "")
                        
                        triple = {
                            "qid": qa_item["id"],
                            "query": query,
                            "pos_id": pos_chunk_id,
                            "pos_text": pos_text,
                            "neg_id": neg_chunk_id,
                            "neg_text": neg_text,
                            "pos_citation": self.corpus_chunks[pos_chunk_id].get("canonical_citation", ""),
                            "neg_citation": self.corpus_chunks[neg_chunk_id].get("canonical_citation", "")
                        }
                        triples.append(triple)
        
        return triples
    
    def create_k_fold_splits(self, k: int = 5) -> List[Tuple[List[Dict], List[Dict]]]:
        """Create k-fold cross-validation splits by section."""
        # Group QA items by section
        section_qa_groups = defaultdict(list)
        
        for qa_item in self.qa_data:
            positive_chunks = self._extract_positive_chunks(qa_item)
            if positive_chunks:
                # Use the section of the first positive chunk
                first_chunk = list(positive_chunks)[0]
                section = self.chunk_to_section.get(first_chunk, "unknown")
                section_qa_groups[section].append(qa_item)
        
        # Create section list and shuffle
        sections = list(section_qa_groups.keys())
        random.shuffle(sections)
        
        # Split sections into k folds
        fold_size = len(sections) // k
        folds = []
        
        for i in range(k):
            start_idx = i * fold_size
            end_idx = start_idx + fold_size if i < k - 1 else len(sections)
            
            test_sections = sections[start_idx:end_idx]
            train_sections = [s for s in sections if s not in test_sections]
            
            # Collect QA items for train and test
            train_qa = []
            test_qa = []
            
            for section in train_sections:
                train_qa.extend(section_qa_groups[section])
            
            for section in test_sections:
                test_qa.extend(section_qa_groups[section])
            
            folds.append((train_qa, test_qa))
        
        return folds
    
    def generate_training_data(self, k_folds: int = 5, num_negatives: int = 3):
        """Generate complete training dataset with k-fold splits."""
        print(f"Generating training data with {k_folds}-fold cross-validation...")
        
        # Create k-fold splits
        folds = self.create_k_fold_splits(k_folds)
        
        # Generate training triples for each fold
        for fold_idx, (train_qa, test_qa) in enumerate(folds):
            print(f"Processing fold {fold_idx + 1}/{k_folds}...")
            print(f"  Train QA items: {len(train_qa)}")
            print(f"  Test QA items: {len(test_qa)}")
            
            # Generate training triples
            self.qa_data = train_qa  # Temporarily set for triple generation
            train_triples = self.create_training_triples(num_negatives)
            
            # Create test queries (no negatives needed)
            test_queries = []
            for qa_item in test_qa:
                positive_chunks = self._extract_positive_chunks(qa_item)
                if positive_chunks:
                    test_queries.append({
                        "qid": qa_item["id"],
                        "query": qa_item["question_user"],
                        "pos_ids": list(positive_chunks),
                        "pos_texts": [self.corpus_chunks[cid].get("text", "") for cid in positive_chunks]
                    })
            
            # Save fold data
            fold_dir = self.output_dir / f"fold_{fold_idx + 1}"
            fold_dir.mkdir(exist_ok=True)
            
            # Save training triples
            train_path = fold_dir / "train_triples.jsonl"
            with train_path.open("w", encoding="utf-8") as f:
                for triple in train_triples:
                    f.write(json.dumps(triple, ensure_ascii=False) + "\n")
            
            # Save test queries
            test_path = fold_dir / "test_queries.jsonl"
            with test_path.open("w", encoding="utf-8") as f:
                for query in test_queries:
                    f.write(json.dumps(query, ensure_ascii=False) + "\n")
            
            print(f"  Generated {len(train_triples)} training triples")
            print(f"  Generated {len(test_queries)} test queries")
            print(f"  Saved to {fold_dir}")
        
        # Restore original QA data
        self.qa_data = self._load_qa_data()
        
        # Generate full training set (no holdout)
        print("\nGenerating full training set...")
        full_triples = self.create_training_triples(num_negatives)
        
        full_path = self.output_dir / "full_train_triples.jsonl"
        with full_path.open("w", encoding="utf-8") as f:
            for triple in full_triples:
                f.write(json.dumps(triple, ensure_ascii=False) + "\n")
        
        print(f"Generated {len(full_triples)} full training triples")
        print(f"Saved to {full_path}")
        
        # Save corpus for training
        corpus_path = self.output_dir / "corpus.jsonl"
        with corpus_path.open("w", encoding="utf-8") as f:
            for chunk in self.corpus_chunks.values():
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
        
        print(f"Saved corpus to {corpus_path}")
        
        return folds, full_triples

def main():
    """Generate training data for dense retrieval."""
    # Paths
    root_dir = Path(__file__).resolve().parent
    corpus_path = root_dir / "data" / "corpus" / "corpus_subsection_v1.jsonl"
    qa_path = root_dir / "data" / "qa" / "pdpa_qa_500.jsonl"
    output_dir = root_dir / "data" / "dense_training"
    
    # Check if files exist
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus not found: {corpus_path}")
    if not qa_path.exists():
        raise FileNotFoundError(f"QA dataset not found: {qa_path}")
    
    # Generate training data
    generator = DenseTrainingDataGenerator(corpus_path, qa_path, output_dir)
    folds, full_triples = generator.generate_training_data(k_folds=5, num_negatives=3)
    
    print(f"\nTraining data generation complete!")
    print(f"Output directory: {output_dir}")
    print(f"Total training triples: {len(full_triples)}")
    print(f"K-fold splits: 5")

if __name__ == "__main__":
    main()