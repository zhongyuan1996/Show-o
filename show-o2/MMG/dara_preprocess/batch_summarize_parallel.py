#!/usr/bin/env python3
"""
Parallel batch processing with vLLM batching.
Processes multiple patients simultaneously while respecting within-patient sequential dependencies.
"""

import argparse
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import sys

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

sys.path.insert(0, str(Path(__file__).parent))
from text_inference_demo import (
    load_visits_from_pickle,
    build_visit_user_content,
    extract_summary_json,
    VisitNarrative,
    record_to_visit,
    _sort_key_for_record,
)


def load_patient_ids(patient_file: Path) -> List[int]:
    """Load patient IDs from JSON or text file."""
    if patient_file.suffix == '.json':
        with open(patient_file) as f:
            data = json.load(f)
            return data.get('patient_ids', [])
    elif patient_file.suffix == '.txt':
        with open(patient_file) as f:
            return [int(line.strip()) for line in f if line.strip()]
    else:
        raise ValueError(f"Unsupported file format: {patient_file.suffix}")


def auto_stratify_patients(
    patient_processors: List['PatientProcessor'],
    max_visit_variance: int = 3,
    min_group_size: int = 50,
) -> List[List['PatientProcessor']]:
    """
    Smart auto-stratification algorithm:
    1. Sort patients by visit count
    2. Group patients with similar visit counts (minimize variance)
    3. Split when variance exceeds threshold
    4. Ensure groups aren't too small
    
    Args:
        patient_processors: List of patients
        max_visit_variance: Maximum visit difference within a group
        min_group_size: Minimum patients per group (avoid tiny groups)
    
    Returns:
        List of patient groups
    """
    if not patient_processors:
        return []
    
    # Sort by visit count
    sorted_patients = sorted(patient_processors, key=lambda p: len(p.visits))
    
    groups = []
    current_group = []
    
    for patient in sorted_patients:
        if not current_group:
            # Start first group
            current_group = [patient]
        else:
            # Check if adding this patient exceeds variance threshold
            group_min = min(len(p.visits) for p in current_group)
            group_max = max(len(p.visits) for p in current_group)
            new_max = max(group_max, len(patient.visits))
            new_range = new_max - group_min
            
            if new_range <= max_visit_variance:
                # Still within variance, add to current group
                current_group.append(patient)
            elif len(current_group) >= min_group_size:
                # Exceeded variance AND current group is large enough
                # → Finalize current group and start new one
                groups.append(current_group)
                current_group = [patient]
            else:
                # Exceeded variance BUT current group is too small
                # → Keep adding to avoid tiny groups
                current_group.append(patient)
    
    # Add final group
    if current_group:
        groups.append(current_group)
    
    return groups


def stratify_patients_by_visit_count(
    patient_processors: List['PatientProcessor'],
    strategy: str = 'quartiles',
    max_visit_variance: int = 3,
    min_group_size: int = 50,
) -> List[List['PatientProcessor']]:
    """
    Group patients by visit count for better batch utilization.
    
    Patients with similar visit counts are processed together to maintain
    high batch sizes throughout each processing group.
    
    Args:
        patient_processors: List of patient processors
        strategy: 'quartiles', 'thirds', 'uniform', or 'auto'
        max_visit_variance: For 'auto' mode, max visit difference within a group
        min_group_size: For 'auto' mode, minimum patients per group
    
    Returns:
        List of patient groups (batches)
    """
    if not patient_processors:
        return []
    
    # Sort by visit count
    sorted_patients = sorted(patient_processors, key=lambda p: len(p.visits))
    visit_counts = [len(p.visits) for p in sorted_patients]
    
    print(f"\n📊 Visit count distribution:")
    print(f"   Min: {min(visit_counts)}, Max: {max(visit_counts)}")
    print(f"   Median: {visit_counts[len(visit_counts)//2]}")
    
    if strategy == 'auto':
        # Use smart auto-stratification
        print(f"   Using auto-stratification (max_variance={max_visit_variance}, min_group_size={min_group_size})")
        groups = auto_stratify_patients(
            patient_processors,
            max_visit_variance=max_visit_variance,
            min_group_size=min_group_size
        )
    
    elif strategy == 'quartiles':
        # Split into 4 groups based on quartiles
        q1_idx = len(sorted_patients) // 4
        q2_idx = len(sorted_patients) // 2
        q3_idx = 3 * len(sorted_patients) // 4
        
        groups = [
            sorted_patients[:q1_idx],      # Q1: shortest
            sorted_patients[q1_idx:q2_idx], # Q2
            sorted_patients[q2_idx:q3_idx], # Q3
            sorted_patients[q3_idx:],       # Q4: longest
        ]
        
    elif strategy == 'thirds':
        # Split into 3 groups
        t1_idx = len(sorted_patients) // 3
        t2_idx = 2 * len(sorted_patients) // 3
        
        groups = [
            sorted_patients[:t1_idx],      # Short
            sorted_patients[t1_idx:t2_idx], # Medium
            sorted_patients[t2_idx:],       # Long
        ]
    
    elif strategy == 'uniform':
        # Try to create groups with similar total visits
        target_visits_per_group = sum(visit_counts) // 4
        groups = []
        current_group = []
        current_visits = 0
        
        for patient in sorted_patients:
            current_group.append(patient)
            current_visits += len(patient.visits)
            
            if current_visits >= target_visits_per_group and len(groups) < 3:
                groups.append(current_group)
                current_group = []
                current_visits = 0
        
        if current_group:
            groups.append(current_group)
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    # Remove empty groups
    groups = [g for g in groups if g]
    
    # Print group statistics with variance
    print(f"\n📦 Created {len(groups)} processing groups:")
    for i, group in enumerate(groups, 1):
        visit_counts_group = [len(p.visits) for p in group]
        visit_min = min(visit_counts_group)
        visit_max = max(visit_counts_group)
        visit_mean = sum(visit_counts_group) / len(visit_counts_group)
        visit_std = (sum((v - visit_mean) ** 2 for v in visit_counts_group) / len(visit_counts_group)) ** 0.5
        total_visits = sum(visit_counts_group)
        
        print(f"   Group {i:2d}: {len(group):5d} patients, "
              f"{visit_min:2d}-{visit_max:2d} visits/patient "
              f"(std: {visit_std:.2f}), {total_visits:5d} total visits")
    
    return groups


def check_gpu_memory() -> Dict[str, Any]:
    """Check GPU memory availability."""
    import subprocess
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free,memory.total", 
             "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True
        )
        gpus = []
        for line in result.stdout.strip().split('\n'):
            idx, free, total = line.split(', ')
            gpus.append({
                'index': int(idx),
                'free_mb': int(free),
                'total_mb': int(total),
                'free_gb': int(free) / 1024
            })
        return {'available': True, 'gpus': gpus}
    except Exception as e:
        return {'available': False, 'error': str(e)}


class PatientProcessor:
    """Tracks state for processing a single patient's visits."""
    
    def __init__(self, subject_id: int, visits: List[VisitNarrative]):
        self.subject_id = subject_id
        self.visits = visits
        self.current_visit_idx = 0
        self.summaries = []
        self.completed = False
        self.error = None
        
    def get_current_visit(self) -> Tuple[VisitNarrative, str]:
        """Get current visit and prior summary."""
        if self.completed or self.current_visit_idx >= len(self.visits):
            return None, None
        
        visit = self.visits[self.current_visit_idx]
        prior = self.summaries[-1] if self.summaries else None
        return visit, prior
    
    def add_summary(self, summary: str):
        """Add summary and advance to next visit."""
        self.summaries.append(summary)
        self.current_visit_idx += 1
        if self.current_visit_idx >= len(self.visits):
            self.completed = True


def _filter_visits_from_records(
    patient_records: List[Dict],
    subject_id: int,
    max_visits: Optional[int] = None,
    truncate_chars: int = 2000,
    min_modalities: int = 1,
) -> Tuple[List[VisitNarrative], int]:
    """
    Filter and convert pre-loaded records to visits for a specific patient.
    This is the in-memory version of load_visits_from_pickle.
    """
    def modality_count(rec: dict) -> int:
        flags = [
            rec.get("has_ehr", False),
            rec.get("has_cxr", False),
            rec.get("has_ecg", False),
            rec.get("has_discharge_note", False),
            rec.get("has_radiology_note", False),
        ]
        return sum(bool(flag) for flag in flags)
    
    # Check if patient meets min_modalities requirement
    counts = [modality_count(r) for r in patient_records]
    if not counts or max(counts) < min_modalities:
        return [], subject_id
    
    # Sort records chronologically
    sorted_records = sorted(patient_records, key=_sort_key_for_record)
    
    # Convert to VisitNarrative objects
    visits = [
        record_to_visit(record, truncate_chars=truncate_chars)
        for record in sorted_records
    ]
    
    # Apply max_visits limit
    if max_visits is not None and max_visits > 0:
        visits = visits[:max_visits]
    
    return visits, subject_id


def load_and_group_records(pickle_path: Path) -> Dict[int, List[Dict]]:
    """Load pickle file once and group by subject_id."""
    import pickle
    from collections import defaultdict
    
    print(f"\n📂 Loading pickle file: {pickle_path}")
    print("⏳ This will take a few minutes (large file ~2.7GB)...")
    
    load_start = time.time()
    with open(pickle_path, 'rb') as f:
        records = pickle.load(f)
    load_time = time.time() - load_start
    
    print(f"✅ Loaded {len(records):,} records in {load_time:.1f}s ({load_time/60:.1f} min)")
    
    # Group by subject_id
    print("📊 Grouping records by patient...")
    grouped = defaultdict(list)
    for record in records:
        grouped[int(record["subject_id"])].append(record)
    
    print(f"✅ Found {len(grouped):,} unique patients")
    return dict(grouped)


def process_patients_parallel(
    subject_ids: List[int],
    pickle_path: Path,
    llm: LLM,
    sampling: SamplingParams,
    output_dir: Path,
    max_visits: int = 5,
    truncate_chars: int = 2000,
    system_prompt: str = "You are an expert clinical summarizer.",
    tokenizer: AutoTokenizer = None,
    batch_size: int = 32,
) -> Dict[str, Any]:
    """
    Process multiple patients in parallel using vLLM batching.
    
    Within each patient, visits are processed sequentially (sliding window).
    Across patients, visits are batched together for parallel inference.
    """
    
    # Load pickle file once (bulk load)
    grouped_records = load_and_group_records(pickle_path)
    
    print(f"\n🔍 Filtering visits for {len(subject_ids)} target patients...")
    filter_start = time.time()
    
    # Filter patients from pre-loaded data (fast, in-memory)
    patient_processors = []
    for idx, subject_id in enumerate(subject_ids, 1):
        try:
            patient_records = grouped_records.get(subject_id, [])
            if not patient_records:
                print(f"⚠️  No records for subject {subject_id}")
                continue
            
            # Filter visits using the same logic as load_visits_from_pickle
            visits, _ = _filter_visits_from_records(
                patient_records,
                subject_id=subject_id,
                max_visits=max_visits,
                truncate_chars=truncate_chars,
                min_modalities=1,
            )
            
            if visits:
                patient_processors.append(PatientProcessor(subject_id, visits))
                # Print progress every 100 patients (fast now!)
                if idx % 100 == 0 or idx == len(subject_ids):
                    elapsed = time.time() - filter_start
                    print(f"  [{idx}/{len(subject_ids)}] Processed {len(patient_processors)} valid patients ({elapsed:.1f}s)")
            else:
                print(f"⚠️  No valid visits for subject {subject_id}")
        except Exception as e:
            print(f"❌ Error processing subject {subject_id}: {e}")
    
    load_time = time.time() - filter_start
    total_visits = sum(len(p.visits) for p in patient_processors)
    print(f"✅ Filtered {len(patient_processors)} patients with {total_visits} total visits in {load_time:.2f}s")
    
    # Dynamic scheduling: collect ALL ready visits each round (maximize batch size)
    round_num = 0
    inference_times = []
    
    inference_start = time.time()
    
    # Track which patients are ready (all have visit 0 ready initially)
    print(f"\n🚀 Starting dynamic batch scheduling...")
    print(f"   Target batch size: {batch_size} visits/round")
    
    while any(not p.completed for p in patient_processors):
        round_num += 1
        round_start = time.time()
        
        # Collect ALL ready visits across all patients (dynamic scheduling)
        batch_data = []
        for processor in patient_processors:
            if not processor.completed:
                visit, prior_summary = processor.get_current_visit()
                if visit:
                    user_content = build_visit_user_content(visit, prior_summary=prior_summary)
                    
                    # Build full prompt with chat template
                    if tokenizer:
                        messages = [
                            {"role": "system", "content": system_prompt.strip()},
                            {"role": "user", "content": user_content}
                        ]
                        prompt = tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True,
                        )
                    else:
                        prompt = f"{system_prompt}\n\n{user_content}"
                    
                    batch_data.append({
                        'processor': processor,
                        'prompt': prompt,
                        'visit': visit,
                    })
        
        if not batch_data:
            break
        
        # Process batch
        prompts = [item['prompt'] for item in batch_data]
        
        # Show batch efficiency
        efficiency = (len(prompts) / batch_size) * 100 if batch_size > 0 else 0
        print(f"\nRound {round_num}: Processing {len(prompts)} visits ({efficiency:.0f}% batch utilization)")
        
        batch_inference_start = time.time()
        outputs = llm.generate(prompts, sampling, use_tqdm=False)
        batch_inference_time = time.time() - batch_inference_start
        inference_times.append(batch_inference_time)
        
        # Distribute results and show visit distribution
        visit_distribution = {}
        for item, output in zip(batch_data, outputs):
            summary = output.outputs[0].text.strip()
            item['processor'].add_summary(summary)
            
            # Track visit distribution
            visit_num = item['processor'].current_visit_idx
            visit_distribution[visit_num] = visit_distribution.get(visit_num, 0) + 1
        
        # Show what was processed
        dist_str = ", ".join([f"V{k}:{v}" for k, v in sorted(visit_distribution.items())[:5]])
        if len(visit_distribution) > 5:
            dist_str += "..."
        
        round_time = time.time() - round_start
        throughput = len(prompts) / batch_inference_time if batch_inference_time > 0 else 0
        print(f"  Distribution: [{dist_str}]")
        print(f"  Completed in {round_time:.2f}s ({batch_inference_time:.2f}s inference, {throughput:.2f} visits/s)")
    
    total_inference_time = time.time() - inference_start
    
    print(f"\n{'='*80}")
    print("Saving results...")
    print(f"{'='*80}")
    
    # Save results
    results = {}
    save_start = time.time()
    
    for processor in patient_processors:
        if processor.error:
            results[processor.subject_id] = {
                'success': False,
                'error': processor.error,
                'num_visits': 0
            }
            continue
        
        # Build export payload
        export_payload = []
        prompt_lengths = []
        
        for visit_idx, (visit, summary) in enumerate(zip(processor.visits, processor.summaries), 1):
            parsed_json, parse_error = extract_summary_json(summary)
            
            prior_summary = processor.summaries[visit_idx - 2] if visit_idx > 1 else None
            input_prompt = build_visit_user_content(visit, prior_summary=prior_summary)
            
            # Calculate token length
            prompt_tokens = len(tokenizer.encode(input_prompt)) if tokenizer else None
            if prompt_tokens:
                prompt_lengths.append(prompt_tokens)
            
            export_payload.append({
                "visit_index": visit_idx,
                "subject_id": visit.subject_id,
                "hadm_id": visit.visit_id,
                "is_virtual": visit.is_virtual,
                "admittime": visit.start_time,
                "dischtime": visit.end_time,
                "modalities_present": [mt.modality for mt in visit.modalities],
                "modalities_text": [
                    {"modality": mt.modality, "text": mt.text}
                    for mt in visit.modalities
                ],
                "input_prompt": input_prompt,
                "input_prompt_tokens": prompt_tokens,
                "prior_summary": prior_summary,
                "raw_summary": summary,
                "summary_json": parsed_json,
                "parse_error": parse_error,
            })
        
        # Save patient output
        output_json = output_dir / f"subject_{processor.subject_id}_summaries.json"
        with output_json.open("w", encoding="utf-8") as f:
            json.dump(export_payload, f, indent=2)
        
        results[processor.subject_id] = {
            'success': True,
            'num_visits': len(processor.visits),
            'output_file': str(output_json),
            'prompt_lengths': {
                'min': min(prompt_lengths) if prompt_lengths else None,
                'max': max(prompt_lengths) if prompt_lengths else None,
                'avg': sum(prompt_lengths) / len(prompt_lengths) if prompt_lengths else None,
            }
        }
        
        print(f"✅ Saved subject {processor.subject_id}: {len(processor.visits)} visits")
    
    save_time = time.time() - save_start
    
    # Calculate statistics
    successful = sum(1 for r in results.values() if r.get('success', False))
    failed = len(results) - successful
    
    return {
        'results': results,
        'summary': {
            'total_patients': len(subject_ids),
            'successful_patients': successful,
            'failed_patients': failed,
            'total_visits': total_visits,
            'total_rounds': round_num,
            'load_time_seconds': load_time,
            'inference_time_seconds': total_inference_time,
            'save_time_seconds': save_time,
            'avg_time_per_patient': total_inference_time / len(patient_processors) if patient_processors else 0,
            'avg_time_per_visit': total_inference_time / total_visits if total_visits > 0 else 0,
            'visits_per_second': total_visits / total_inference_time if total_inference_time > 0 else 0,
            'avg_batch_size': total_visits / round_num if round_num > 0 else 0,
        }
    }


def process_patients_stratified(
    subject_ids: List[int],
    pickle_path: Path,
    llm: LLM,
    sampling: SamplingParams,
    output_dir: Path,
    max_visits: int = 5,
    truncate_chars: int = 2000,
    system_prompt: str = "You are an expert clinical summarizer.",
    tokenizer: AutoTokenizer = None,
    batch_size: int = 32,
    stratify_strategy: str = 'quartiles',
    max_visit_variance: int = 3,
    min_group_size: int = 50,
) -> Dict[str, Any]:
    """
    Process patients with stratification by visit count for better GPU utilization.
    """
    
    # Load pickle file once (bulk load)
    grouped_records = load_and_group_records(pickle_path)
    
    print(f"\n🔍 Filtering visits for {len(subject_ids)} target patients...")
    filter_start = time.time()
    
    # Filter ALL patients from pre-loaded data (fast, in-memory)
    all_patient_processors = []
    for idx, subject_id in enumerate(subject_ids, 1):
        try:
            patient_records = grouped_records.get(subject_id, [])
            if not patient_records:
                continue
            
            visits, _ = _filter_visits_from_records(
                patient_records,
                subject_id=subject_id,
                max_visits=max_visits,
                truncate_chars=truncate_chars,
                min_modalities=1,
            )
            
            if visits:
                all_patient_processors.append(PatientProcessor(subject_id, visits))
                if idx % 100 == 0 or idx == len(subject_ids):
                    elapsed = time.time() - filter_start
                    print(f"  [{idx}/{len(subject_ids)}] Processed {len(all_patient_processors)} valid patients ({elapsed:.1f}s)")
        except Exception as e:
            print(f"❌ Error processing subject {subject_id}: {e}")
    
    load_time = time.time() - filter_start
    total_visits = sum(len(p.visits) for p in all_patient_processors)
    print(f"✅ Filtered {len(all_patient_processors)} patients with {total_visits} total visits in {load_time:.2f}s")
    
    # Stratify patients into groups
    patient_groups = stratify_patients_by_visit_count(
        all_patient_processors,
        strategy=stratify_strategy,
        max_visit_variance=max_visit_variance,
        min_group_size=min_group_size,
    )
    
    # Process each group sequentially
    all_results = {}
    total_inference_time = 0
    total_rounds = 0
    total_save_time = 0
    
    for group_idx, patient_group in enumerate(patient_groups, 1):
        print(f"\n{'='*80}")
        print(f"PROCESSING GROUP {group_idx}/{len(patient_groups)}: {len(patient_group)} patients")
        print(f"{'='*80}")
        
        group_results = _process_patient_group(
            patient_processors=patient_group,
            llm=llm,
            sampling=sampling,
            output_dir=output_dir,
            system_prompt=system_prompt,
            tokenizer=tokenizer,
            batch_size=batch_size,
        )
        
        # Aggregate results
        all_results.update(group_results['results'])
        total_inference_time += group_results['summary']['inference_time_seconds']
        total_rounds += group_results['summary']['total_rounds']
        total_save_time += group_results['summary']['save_time_seconds']
        
        # Print group summary
        gs = group_results['summary']
        print(f"\n✅ Group {group_idx} completed:")
        print(f"   Patients: {gs['successful_patients']}/{gs['total_patients']}")
        print(f"   Visits: {gs['total_visits']}")
        print(f"   Rounds: {gs['total_rounds']}")
        print(f"   Time: {gs['inference_time_seconds']:.2f}s ({gs['inference_time_seconds']/60:.2f} min)")
        print(f"   Throughput: {gs['visits_per_second']:.3f} visits/s")
    
    # Calculate aggregate statistics
    successful = sum(1 for r in all_results.values() if r.get('success', False))
    failed = len(all_results) - successful
    
    return {
        'results': all_results,
        'summary': {
            'total_patients': len(subject_ids),
            'successful_patients': successful,
            'failed_patients': failed,
            'total_visits': total_visits,
            'total_rounds': total_rounds,
            'load_time_seconds': load_time,
            'inference_time_seconds': total_inference_time,
            'save_time_seconds': total_save_time,
            'avg_time_per_patient': total_inference_time / len(all_patient_processors) if all_patient_processors else 0,
            'avg_time_per_visit': total_inference_time / total_visits if total_visits > 0 else 0,
            'visits_per_second': total_visits / total_inference_time if total_inference_time > 0 else 0,
            'avg_batch_size': total_visits / total_rounds if total_rounds > 0 else 0,
            'num_groups': len(patient_groups),
        }
    }


def _process_patient_group(
    patient_processors: List['PatientProcessor'],
    llm: LLM,
    sampling: SamplingParams,
    output_dir: Path,
    system_prompt: str,
    tokenizer: AutoTokenizer,
    batch_size: int,
) -> Dict[str, Any]:
    """Process a single group of patients."""
    
    total_visits = sum(len(p.visits) for p in patient_processors)
    
    # Process visits in rounds (batching across patients)
    round_num = 0
    inference_times = []
    
    inference_start = time.time()
    
    print(f"\n🚀 Starting dynamic batch scheduling...")
    print(f"   Target batch size: {batch_size} visits/round")
    
    while any(not p.completed for p in patient_processors):
        round_num += 1
        round_start = time.time()
        
        # Collect ALL ready visits across all patients
        batch_data = []
        for processor in patient_processors:
            if not processor.completed:
                visit, prior_summary = processor.get_current_visit()
                if visit:
                    user_content = build_visit_user_content(visit, prior_summary=prior_summary)
                    
                    if tokenizer:
                        messages = [
                            {"role": "system", "content": system_prompt.strip()},
                            {"role": "user", "content": user_content}
                        ]
                        prompt = tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True,
                        )
                    else:
                        prompt = f"{system_prompt}\n\n{user_content}"
                    
                    batch_data.append({
                        'processor': processor,
                        'prompt': prompt,
                        'visit': visit,
                    })
        
        if not batch_data:
            break
        
        # Process batch
        prompts = [item['prompt'] for item in batch_data]
        
        # Show batch efficiency
        efficiency = (len(prompts) / batch_size) * 100 if batch_size > 0 else 0
        print(f"\nRound {round_num}: Processing {len(prompts)} visits ({efficiency:.0f}% batch utilization)")
        
        batch_inference_start = time.time()
        outputs = llm.generate(prompts, sampling, use_tqdm=False)
        batch_inference_time = time.time() - batch_inference_start
        inference_times.append(batch_inference_time)
        
        # Distribute results and show visit distribution
        visit_distribution = {}
        for item, output in zip(batch_data, outputs):
            summary = output.outputs[0].text.strip()
            item['processor'].add_summary(summary)
            
            # Track visit distribution
            visit_num = item['processor'].current_visit_idx
            visit_distribution[visit_num] = visit_distribution.get(visit_num, 0) + 1
        
        # Show what was processed
        dist_str = ", ".join([f"V{k}:{v}" for k, v in sorted(visit_distribution.items())[:5]])
        if len(visit_distribution) > 5:
            dist_str += "..."
        
        round_time = time.time() - round_start
        throughput = len(prompts) / batch_inference_time if batch_inference_time > 0 else 0
        print(f"  Distribution: [{dist_str}]")
        print(f"  Completed in {round_time:.2f}s ({batch_inference_time:.2f}s inference, {throughput:.2f} visits/s)")
    
    total_inference_time = time.time() - inference_start
    
    print(f"\n{'='*80}")
    print("Saving results...")
    print(f"{'='*80}")
    
    # Save results
    results = {}
    save_start = time.time()
    
    for processor in patient_processors:
        if processor.error:
            results[processor.subject_id] = {
                'success': False,
                'error': processor.error,
                'num_visits': 0
            }
            continue
        
        # Build export payload
        export_payload = []
        prompt_lengths = []
        
        for visit_idx, (visit, summary) in enumerate(zip(processor.visits, processor.summaries), 1):
            parsed_json, parse_error = extract_summary_json(summary)
            
            prior_summary = processor.summaries[visit_idx - 2] if visit_idx > 1 else None
            input_prompt = build_visit_user_content(visit, prior_summary=prior_summary)
            
            # Calculate token length
            prompt_tokens = len(tokenizer.encode(input_prompt)) if tokenizer else None
            if prompt_tokens:
                prompt_lengths.append(prompt_tokens)
            
            export_payload.append({
                "visit_index": visit_idx,
                "subject_id": visit.subject_id,
                "hadm_id": visit.visit_id,
                "is_virtual": visit.is_virtual,
                "admittime": visit.start_time,
                "dischtime": visit.end_time,
                "modalities_present": [mt.modality for mt in visit.modalities],
                "modalities_text": [
                    {"modality": mt.modality, "text": mt.text}
                    for mt in visit.modalities
                ],
                "input_prompt": input_prompt,
                "input_prompt_tokens": prompt_tokens,
                "prior_summary": prior_summary,
                "raw_summary": summary,
                "summary_json": parsed_json,
                "parse_error": parse_error,
            })
        
        # Save patient output
        output_json = output_dir / f"subject_{processor.subject_id}_summaries.json"
        with output_json.open("w", encoding="utf-8") as f:
            json.dump(export_payload, f, indent=2)
        
        results[processor.subject_id] = {
            'success': True,
            'num_visits': len(processor.visits),
            'output_file': str(output_json),
            'prompt_lengths': {
                'min': min(prompt_lengths) if prompt_lengths else None,
                'max': max(prompt_lengths) if prompt_lengths else None,
                'avg': sum(prompt_lengths) / len(prompt_lengths) if prompt_lengths else None,
            }
        }
    
    save_time = time.time() - save_start
    
    # Calculate statistics
    successful = sum(1 for r in results.values() if r.get('success', False))
    failed = len(results) - successful
    
    return {
        'results': results,
        'summary': {
            'total_patients': len(patient_processors),
            'successful_patients': successful,
            'failed_patients': failed,
            'total_visits': total_visits,
            'total_rounds': round_num,
            'inference_time_seconds': total_inference_time,
            'save_time_seconds': save_time,
            'avg_time_per_patient': total_inference_time / len(patient_processors) if patient_processors else 0,
            'avg_time_per_visit': total_inference_time / total_visits if total_visits > 0 else 0,
            'visits_per_second': total_visits / total_inference_time if total_inference_time > 0 else 0,
            'avg_batch_size': total_visits / round_num if round_num > 0 else 0,
        }
    }


def main():
    parser = argparse.ArgumentParser(
        description="Parallel batch processing with vLLM batching"
    )
    parser.add_argument(
        "--patient-file",
        type=Path,
        required=True,
        help="JSON or text file containing patient IDs"
    )
    parser.add_argument(
        "--pickle-path",
        type=Path,
        default=Path("output/v1/matching_results.pkl"),
        help="Path to matching_results.pkl"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/summaries_parallel"),
        help="Output directory"
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-72B-Instruct",
        help="Model name"
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=2,
        help="Tensor parallelism"
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=8192,
        help="Max context length"
    )
    parser.add_argument(
        "--max-visits",
        type=int,
        default=22,
        help="Max visits per patient (95th percentile)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Maximum batch size for parallel inference"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=800,
        help="Max output tokens"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of patients"
    )
    parser.add_argument(
        "--stratify",
        type=str,
        choices=['none', 'quartiles', 'thirds', 'uniform', 'auto'],
        default='auto',
        help="Stratify patients by visit count for better batch utilization (default: auto)"
    )
    parser.add_argument(
        "--max-visit-variance",
        type=int,
        default=3,
        help="For auto stratification: max visit difference within a group (default: 3)"
    )
    parser.add_argument(
        "--min-group-size",
        type=int,
        default=50,
        help="For auto stratification: minimum patients per group (default: 50)"
    )
    
    args = parser.parse_args()
    
    # Check GPU
    print("Checking GPU...")
    gpu_info = check_gpu_memory()
    if gpu_info['available']:
        for gpu in gpu_info['gpus']:
            print(f"  GPU {gpu['index']}: {gpu['free_gb']:.1f} GB free")
    
    # Load patients
    print(f"\nLoading patient IDs from: {args.patient_file}")
    all_ids = load_patient_ids(args.patient_file)
    patient_ids = all_ids[:args.limit] if args.limit else all_ids
    print(f"Processing {len(patient_ids)} patients")
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print(f"\n{'='*80}")
    print(f"Loading {args.model}...")
    print(f"{'='*80}")
    
    model_start = time.time()
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
    )
    model_load_time = time.time() - model_start
    print(f"✅ Model loaded in {model_load_time:.2f}s ({model_load_time/60:.2f} min)")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=0.9,
        max_tokens=args.max_output_tokens,
    )
    
    # Process with optional stratification
    print(f"\n{'='*80}")
    print(f"PARALLEL PROCESSING: {len(patient_ids)} patients")
    if args.stratify != 'none':
        print(f"STRATIFICATION: {args.stratify}")
    print(f"{'='*80}")
    
    if args.stratify == 'none':
        # Process all patients together (original behavior)
        results = process_patients_parallel(
            subject_ids=patient_ids,
            pickle_path=args.pickle_path,
            llm=llm,
            sampling=sampling,
            output_dir=args.output_dir,
            max_visits=args.max_visits,
            system_prompt="You are an expert clinical summarizer.",
            tokenizer=tokenizer,
            batch_size=args.batch_size,
        )
    else:
        # Load and stratify patients first
        results = process_patients_stratified(
            subject_ids=patient_ids,
            pickle_path=args.pickle_path,
            llm=llm,
            sampling=sampling,
            output_dir=args.output_dir,
            max_visits=args.max_visits,
            system_prompt="You are an expert clinical summarizer.",
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            stratify_strategy=args.stratify,
            max_visit_variance=args.max_visit_variance,
            min_group_size=args.min_group_size,
        )
    
    # Summary
    summary = results['summary']
    print(f"\n{'='*80}")
    print("RESULTS")
    print(f"{'='*80}")
    print(f"Patients: {summary['successful_patients']}/{summary['total_patients']}")
    print(f"Total visits: {summary['total_visits']}")
    print(f"Processing rounds: {summary['total_rounds']}")
    print(f"Avg batch size: {summary['avg_batch_size']:.1f} visits/round")
    print(f"\nTiming:")
    print(f"  Model load: {model_load_time:.2f}s")
    print(f"  Data load: {summary['load_time_seconds']:.2f}s")
    print(f"  Inference: {summary['inference_time_seconds']:.2f}s ({summary['inference_time_seconds']/60:.2f} min)")
    print(f"  Save: {summary['save_time_seconds']:.2f}s")
    print(f"  Throughput: {summary['visits_per_second']:.3f} visits/second")
    print(f"\nProjection for 1000 patients:")
    est_time = summary['avg_time_per_patient'] * 1000
    print(f"  {est_time:.2f}s ({est_time/60:.2f} min, {est_time/3600:.2f} hours)")
    
    # Save benchmark
    results_file = args.output_dir / "parallel_benchmark_results.json"
    
    # Convert Path objects to strings for JSON serialization
    config_dict = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    
    with open(results_file, 'w') as f:
        json.dump({
            'config': config_dict,
            'model_load_time': model_load_time,
            'summary': summary,
            'gpu_info': gpu_info,
        }, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()

