#!/usr/bin/env python3
"""
Filter patients that have all 5 modalities (EHR, CXR, ECG, discharge note, radiology note)
in their medical history across all visits.
"""

import pickle
import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set
import random
import numpy as np


def filter_patients_with_all_modalities(
    pickle_path: Path,
    min_visits: int = 2,
) -> Dict[int, Dict]:
    """
    Filter patients that have all 5 modalities across their visits.
    
    Returns:
        Dictionary mapping subject_id to patient info
    """
    print(f"Loading data from: {pickle_path}")
    with open(pickle_path, 'rb') as f:
        records = pickle.load(f)
    
    print(f"Total records: {len(records)}")
    
    # Group by patient
    patients = defaultdict(list)
    for record in records:
        subject_id = record.get('subject_id')
        if subject_id:
            patients[subject_id].append(record)
    
    print(f"Total unique patients: {len(patients)}")
    
    # Filter for patients with all 5 modalities
    modality_flags = ['has_ehr', 'has_cxr', 'has_ecg', 'has_discharge_note', 'has_radiology_note']
    
    filtered_patients = {}
    
    for subject_id, visits in patients.items():
        if len(visits) < min_visits:
            continue
        
        # Check if patient has all 5 modalities across any of their visits
        patient_modalities = set()
        for visit in visits:
            for flag in modality_flags:
                if visit.get(flag, False):
                    patient_modalities.add(flag)
        
        # If patient has all 5 modalities
        if len(patient_modalities) == 5:
            filtered_patients[subject_id] = {
                'subject_id': subject_id,
                'num_visits': len(visits),
                'modalities': list(patient_modalities),
                'visits': visits,
            }
    
    print(f"\nFiltered patients with all 5 modalities: {len(filtered_patients)}")
    
    # Statistics
    visit_counts = [info['num_visits'] for info in filtered_patients.values()]
    print(f"Visit statistics:")
    print(f"  Min visits: {min(visit_counts)}")
    print(f"  Max visits: {max(visit_counts)}")
    print(f"  Avg visits: {sum(visit_counts) / len(visit_counts):.2f}")
    print(f"  Median (50th): {np.median(visit_counts):.0f}")
    print(f"  95th percentile: {np.percentile(visit_counts, 95):.0f}")
    
    return filtered_patients


def main():
    parser = argparse.ArgumentParser(description="Filter patients with all 5 modalities")
    parser.add_argument(
        "--pickle-path",
        type=Path,
        default=Path("output/v1/matching_results.pkl"),
        help="Path to matching_results.pkl"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/filtered_patients"),
        help="Directory to save filtered patient IDs"
    )
    parser.add_argument(
        "--min-visits",
        type=int,
        default=2,
        help="Minimum number of visits required"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=1000,
        help="Number of patients to sample for batch processing"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter patients
    filtered_patients = filter_patients_with_all_modalities(
        pickle_path=args.pickle_path,
        min_visits=args.min_visits,
    )
    
    # Save all filtered patient IDs
    all_ids_file = args.output_dir / "all_multimodal_patients.json"
    all_patient_ids = sorted(filtered_patients.keys())
    
    with open(all_ids_file, 'w') as f:
        json.dump({
            'num_patients': len(all_patient_ids),
            'patient_ids': all_patient_ids,
            'filter_criteria': {
                'all_5_modalities': True,
                'min_visits': args.min_visits,
            },
            'statistics': {
                'min_visits': min(info['num_visits'] for info in filtered_patients.values()),
                'max_visits': max(info['num_visits'] for info in filtered_patients.values()),
                'avg_visits': sum(info['num_visits'] for info in filtered_patients.values()) / len(filtered_patients),
                'median_visits': float(np.median([info['num_visits'] for info in filtered_patients.values()])),
                'p75_visits': float(np.percentile([info['num_visits'] for info in filtered_patients.values()], 75)),
                'p90_visits': float(np.percentile([info['num_visits'] for info in filtered_patients.values()], 90)),
                'p95_visits': float(np.percentile([info['num_visits'] for info in filtered_patients.values()], 95)),
                'p99_visits': float(np.percentile([info['num_visits'] for info in filtered_patients.values()], 99)),
            }
        }, f, indent=2)
    
    print(f"\nSaved all {len(all_patient_ids)} patient IDs to: {all_ids_file}")
    
    # Sample subset for batch processing
    if args.sample_size > 0 and args.sample_size < len(all_patient_ids):
        random.seed(args.seed)
        sampled_ids = sorted(random.sample(all_patient_ids, args.sample_size))
        
        sample_file = args.output_dir / f"sampled_{args.sample_size}_patients.json"
        with open(sample_file, 'w') as f:
            json.dump({
                'num_patients': len(sampled_ids),
                'patient_ids': sampled_ids,
                'sample_size': args.sample_size,
                'seed': args.seed,
                'source': str(all_ids_file),
            }, f, indent=2)
        
        print(f"Sampled {len(sampled_ids)} patients to: {sample_file}")
        
        # Also save as simple text file for easy use
        sample_txt_file = args.output_dir / f"sampled_{args.sample_size}_patients.txt"
        with open(sample_txt_file, 'w') as f:
            for pid in sampled_ids:
                f.write(f"{pid}\n")
        
        print(f"Patient IDs also saved to: {sample_txt_file}")
    
    # Save detailed info for sampled patients
    if args.sample_size > 0 and args.sample_size < len(all_patient_ids):
        detailed_file = args.output_dir / f"sampled_{args.sample_size}_patients_detailed.json"
        sampled_details = {
            str(pid): {
                'num_visits': filtered_patients[pid]['num_visits'],
                'modalities': filtered_patients[pid]['modalities'],
            }
            for pid in sampled_ids
        }
        
        with open(detailed_file, 'w') as f:
            json.dump({
                'num_patients': len(sampled_ids),
                'patients': sampled_details,
                'summary_statistics': {
                    'total_visits': sum(d['num_visits'] for d in sampled_details.values()),
                    'avg_visits_per_patient': sum(d['num_visits'] for d in sampled_details.values()) / len(sampled_details),
                }
            }, f, indent=2)
        
        print(f"Detailed info saved to: {detailed_file}")


if __name__ == "__main__":
    main()

