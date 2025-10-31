#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
create_stratified_splits.py — Create stratified 80/10/10 splits for dense retrieval training
- Maintains representative distribution of sections and qa_types
- Creates train (80%), validation (10%), and test (10%) splits
- Ensures balanced representation across all splits
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict, Counter
import numpy as np

class StratifiedSplitter:
    """Create stratified splits maintaining section and qa_type distribution."""
    
    def __init__(self, full_train_triples_path: Path, qa_dataset_path: Path, output_dir: Path):
        """Initialize with paths to training data and QA dataset."""
        self.full_train_triples_path = Path(full_train_triples_path)
        self.qa_dataset_path = Path(qa_dataset_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        self.training_triples = self._load_training_triples()
        self.qa_data = self._load_qa_data()
        
        # Create mappings
        self.qid_to_qa_info = {qa['id']: qa for qa in self.qa_data}
        
    def _load_training_triples(self) -> List[Dict]:
        """Load training triples from JSONL file."""
        triples = []
        with self.full_train_triples_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    triples.append(json.loads(line))
        return triples
    
    def _load_qa_data(self) -> List[Dict]:
        """Load QA dataset."""
        qa_data = []
        with self.qa_dataset_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    qa_item = json.loads(line)
                    if 'qa_type' not in qa_item:
                        qa_item['qa_type'] = 'unknown'
                    qa_data.append(qa_item)
        return qa_data
    
    def _get_qa_info(self, qid: str) -> Dict:
        """Get QA information for a given QID."""
        return self.qid_to_qa_info.get(qid, {
            'qa_type': 'unknown',
            'part': '',
            'canonical_sections': [],
            'difficulty': 'medium'
        })
    
    def _extract_section_key(self, qa_info: Dict) -> str:
        """Extract section key from QA info."""
        part = qa_info.get('part', '')
        canonical_sections = qa_info.get('canonical_sections', [])
        if canonical_sections:
            # Use first canonical section as primary key
            return f"Part{part}_{canonical_sections[0]}"
        return f"Part{part}_unknown"
    
    def create_stratified_splits(self, train_ratio: float = 0.8, val_ratio: float = 0.1, test_ratio: float = 0.1):
        """Create stratified splits maintaining section and qa_type distribution."""
        print(f"Creating stratified splits: {train_ratio:.0%}/{val_ratio:.0%}/{test_ratio:.0%}")
        
        # Group triples by QID to maintain query-level splits
        qid_to_triples = defaultdict(list)
        for triple in self.training_triples:
            qid_to_triples[triple['qid']].append(triple)
        
        # Get unique QIDs and their metadata
        unique_qids = list(qid_to_triples.keys())
        qid_metadata = []
        
        for qid in unique_qids:
            qa_info = self._get_qa_info(qid)
            section_key = self._extract_section_key(qa_info)
            qa_type = qa_info.get('qa_type', 'unknown')
            
            qid_metadata.append({
                'qid': qid,
                'section_key': section_key,
                'qa_type': qa_type,
                'part': qa_info.get('part', ''),
                'num_triples': len(qid_to_triples[qid])
            })
        
        print(f"Total unique queries: {len(unique_qids)}")
        print(f"Total training triples: {len(self.training_triples)}")
        
        # Analyze distribution
        self._analyze_distribution(qid_metadata)
        
        # Create stratified splits
        train_qids, val_qids, test_qids = self._stratified_split(qid_metadata, train_ratio, val_ratio, test_ratio)
        
        # Generate splits
        train_triples = []
        val_triples = []
        test_triples = []
        
        for qid in train_qids:
            train_triples.extend(qid_to_triples[qid])
        
        for qid in val_qids:
            val_triples.extend(qid_to_triples[qid])
        
        for qid in test_qids:
            test_triples.extend(qid_to_triples[qid])
        
        # Shuffle within each split
        random.shuffle(train_triples)
        random.shuffle(val_triples)
        random.shuffle(test_triples)
        
        print(f"\nSplit sizes:")
        print(f"  Train: {len(train_triples)} triples from {len(train_qids)} queries")
        print(f"  Val:   {len(val_triples)} triples from {len(val_qids)} queries")
        print(f"  Test:  {len(test_triples)} triples from {len(test_qids)} queries")
        
        # Save splits
        self._save_splits(train_triples, val_triples, test_triples)
        
        # Verify stratification
        self._verify_stratification(train_qids, val_qids, test_qids, qid_metadata)
        
        return train_triples, val_triples, test_triples
    
    def _analyze_distribution(self, qid_metadata: List[Dict]):
        """Analyze the distribution of sections and qa_types."""
        section_counts = Counter(item['section_key'] for item in qid_metadata)
        qa_type_counts = Counter(item['qa_type'] for item in qid_metadata)
        part_counts = Counter(item['part'] for item in qid_metadata)
        
        print(f"\nDistribution Analysis:")
        print(f"  Sections: {len(section_counts)} unique sections")
        print(f"  QA Types: {dict(qa_type_counts)}")
        print(f"  Parts: {dict(part_counts)}")
        
        # Show top sections
        print(f"  Top 10 sections:")
        for section, count in section_counts.most_common(10):
            print(f"    {section}: {count}")
    
    def _stratified_split(self, qid_metadata: List[Dict], train_ratio: float, val_ratio: float, test_ratio: float) -> Tuple[List[str], List[str], List[str]]:
        """Perform stratified splitting based on section and qa_type."""
        # Group by qa_type only for better balance (sections are too granular)
        strata = defaultdict(list)
        for item in qid_metadata:
            stratum_key = item['qa_type']
            strata[stratum_key].append(item['qid'])
        
        train_qids = []
        val_qids = []
        test_qids = []
        
        print(f"\nStratified splitting across {len(strata)} qa_type strata...")
        
        for stratum_key, qids in strata.items():
            # Shuffle QIDs within stratum
            random.shuffle(qids)
            
            n_total = len(qids)
            n_train = int(n_total * train_ratio)
            n_val = int(n_total * val_ratio)
            n_test = n_total - n_train - n_val  # Remainder goes to test
            
            # Ensure minimum allocation for each split if stratum is large enough
            if n_total >= 3:
                n_train = max(1, n_train)
                n_val = max(1, n_val)
                n_test = max(1, n_test)
                
                # Adjust if total exceeds available
                total_allocated = n_train + n_val + n_test
                if total_allocated > n_total:
                    # Reduce proportionally
                    excess = total_allocated - n_total
                    if n_train > 1:
                        n_train -= min(excess, n_train - 1)
                        excess = max(0, excess - (n_train - 1))
                    if excess > 0 and n_val > 1:
                        n_val -= min(excess, n_val - 1)
                        excess = max(0, excess - (n_val - 1))
                    if excess > 0 and n_test > 1:
                        n_test -= excess
                    
                    # Recalculate to ensure we use all samples
                    n_test = n_total - n_train - n_val
            elif n_total == 2:
                # For small strata, put one in train and one in test
                n_train = 1
                n_val = 0
                n_test = 1
            else:
                # Single item goes to train
                n_train = 1
                n_val = 0
                n_test = 0
            
            # Split the QIDs
            train_qids.extend(qids[:n_train])
            val_qids.extend(qids[n_train:n_train + n_val])
            test_qids.extend(qids[n_train + n_val:n_train + n_val + n_test])
            
            print(f"  {stratum_key}: {n_total} -> {n_train}/{n_val}/{n_test}")
        
        return train_qids, val_qids, test_qids
    
    def _save_splits(self, train_triples: List[Dict], val_triples: List[Dict], test_triples: List[Dict]):
        """Save the splits to files."""
        splits_dir = self.output_dir / "stratified_splits"
        splits_dir.mkdir(exist_ok=True)
        
        # Save training split
        train_path = splits_dir / "train_triples.jsonl"
        with train_path.open("w", encoding="utf-8") as f:
            for triple in train_triples:
                f.write(json.dumps(triple, ensure_ascii=False) + "\n")
        
        # Save validation split
        val_path = splits_dir / "val_triples.jsonl"
        with val_path.open("w", encoding="utf-8") as f:
            for triple in val_triples:
                f.write(json.dumps(triple, ensure_ascii=False) + "\n")
        
        # Save test split
        test_path = splits_dir / "test_triples.jsonl"
        with test_path.open("w", encoding="utf-8") as f:
            for triple in test_triples:
                f.write(json.dumps(triple, ensure_ascii=False) + "\n")
        
        print(f"\nSplits saved to {splits_dir}:")
        print(f"  Train: {train_path}")
        print(f"  Val:   {val_path}")
        print(f"  Test:  {test_path}")
    
    def _verify_stratification(self, train_qids: List[str], val_qids: List[str], test_qids: List[str], qid_metadata: List[Dict]):
        """Verify that stratification was successful."""
        print(f"\nStratification Verification:")
        
        # Create lookup for QID metadata
        qid_to_meta = {item['qid']: item for item in qid_metadata}
        
        # Analyze qa_type distribution across splits
        for split_name, qids in [("Train", train_qids), ("Val", val_qids), ("Test", test_qids)]:
            qa_type_counts = Counter(qid_to_meta[qid]['qa_type'] for qid in qids)
            total = len(qids)
            
            print(f"  {split_name} ({total} queries):")
            for qa_type, count in sorted(qa_type_counts.items()):
                percentage = (count / total) * 100 if total > 0 else 0
                print(f"    {qa_type}: {count} ({percentage:.1f}%)")

def main():
    """Create stratified splits for dense retrieval training."""
    # Paths
    root_dir = Path(__file__).resolve().parent
    full_train_triples_path = root_dir / "data" / "dense_training" / "full_train_triples.jsonl"
    qa_dataset_path = root_dir / "data" / "qa" / "pdpa_qa_500.jsonl"
    output_dir = root_dir / "data" / "dense_training"
    
    # Check if files exist
    if not full_train_triples_path.exists():
        raise FileNotFoundError(f"Training triples not found: {full_train_triples_path}")
    if not qa_dataset_path.exists():
        raise FileNotFoundError(f"QA dataset not found: {qa_dataset_path}")
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Create stratified splits
    splitter = StratifiedSplitter(full_train_triples_path, qa_dataset_path, output_dir)
    train_triples, val_triples, test_triples = splitter.create_stratified_splits(
        train_ratio=0.8, val_ratio=0.1, test_ratio=0.1
    )
    
    print(f"\nStratified splits created successfully!")
    print(f"Use these splits for:")
    print(f"  - Training: 80% for dense retriever fine-tuning")
    print(f"  - Validation: 10% for hyperparameter tuning in hybrid retriever")
    print(f"  - Test: 10% for final model evaluation")

if __name__ == "__main__":
    main()