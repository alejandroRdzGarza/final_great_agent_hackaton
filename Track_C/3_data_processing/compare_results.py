"""
Compare attack_successful values between validation_results.json and multi_turn_results.json.

Specifically tracks:
- True -> False transitions (regressions - we don't want these)
- False -> True transitions (we don't want these either)
"""

import json
import os
from collections import defaultdict
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def load_json(filepath):
    """Load JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_attack_successful(result):
    """Extract attack_successful from result, handling different structures."""
    if isinstance(result, dict):
        # Check if it's directly in the result
        if 'attack_successful' in result:
            return result['attack_successful']
        # Check if it's in judgment
        if 'judgment' in result and isinstance(result['judgment'], dict):
            return result['judgment'].get('attack_successful')
    return None

def get_key(result):
    """Get a unique key for matching results (prefer question_id, fallback to row_idx)."""
    if isinstance(result, dict):
        if 'question_id' in result and result['question_id'] is not None:
            return ('question_id', result['question_id'])
        if 'row_idx' in result and result['row_idx'] is not None:
            return ('row_idx', result['row_idx'])
    return None

def compare_results(validation_file, multi_turn_file):
    """Compare attack_successful values between two result files."""
    
    print("Loading files...")
    validation_results = load_json(validation_file)
    multi_turn_results = load_json(multi_turn_file)
    
    print(f"Validation results: {len(validation_results)} entries")
    print(f"Multi-turn results: {len(multi_turn_results)} entries")
    
    # Create lookup dictionaries
    validation_lookup = {}
    for result in validation_results:
        key = get_key(result)
        if key:
            validation_lookup[key] = result
    
    multi_turn_lookup = {}
    for result in multi_turn_results:
        key = get_key(result)
        if key:
            multi_turn_lookup[key] = result
    
    print(f"\nValidation entries with keys: {len(validation_lookup)}")
    print(f"Multi-turn entries with keys: {len(multi_turn_lookup)}")
    
    # Find common keys
    common_keys = set(validation_lookup.keys()) & set(multi_turn_lookup.keys())
    print(f"Common entries (matched): {len(common_keys)}")
    
    # Track transitions
    true_to_false = []  # Regressions - we don't want these
    false_to_true = []  # We don't want these either
    true_to_true = []
    false_to_false = []
    missing_in_validation = []
    missing_in_multi_turn = []
    
    # Collect labels for sklearn classification report
    # OLD (multi_turn) = ground truth, NEW (validation) = predictions
    y_true = []  # OLD (multi_turn) - ground truth
    y_pred = []  # NEW (validation) - predictions
    valid_keys = []  # keys that have valid labels
    
    # Compare common entries
    for key in common_keys:
        val_result = validation_lookup[key]  # NEW
        mt_result = multi_turn_lookup[key]   # OLD
        
        val_attack = get_attack_successful(val_result)  # NEW
        mt_attack = get_attack_successful(mt_result)    # OLD
        
        if val_attack is None:
            missing_in_validation.append(key)
            continue
        if mt_attack is None:
            missing_in_multi_turn.append(key)
            continue
        
        # Collect for classification report (OLD as ground truth, NEW as predictions)
        y_true.append(mt_attack)  # OLD
        y_pred.append(val_attack)  # NEW
        valid_keys.append(key)
        
        # Track transitions: OLD -> NEW
        # OLD success -> NEW failure = REGRESSION (BAD)
        if mt_attack is True and val_attack is False:
            true_to_false.append({
                'key': key,
                'old_multi_turn': mt_attack,
                'new_validation': val_attack,
                'question_id': val_result.get('question_id'),
                'row_idx': val_result.get('row_idx'),
                'submission_message': val_result.get('submission_message', ''),
                'old_conversation': mt_result.get('conversation', []),
                'old_final_messages': mt_result.get('final_messages', []),
                'old_judgment': mt_result.get('judgment', {}),
                'new_conversation': val_result.get('conversation', []),
                'new_final_messages': val_result.get('final_messages', []),
                'new_judgment': val_result.get('judgment', {})
            })
        # OLD failure -> NEW success = NEW ATTACK (BAD)
        elif mt_attack is False and val_attack is True:
            false_to_true.append({
                'key': key,
                'old_multi_turn': mt_attack,
                'new_validation': val_attack,
                'question_id': val_result.get('question_id'),
                'row_idx': val_result.get('row_idx'),
                'submission_message': val_result.get('submission_message', ''),
                'old_conversation': mt_result.get('conversation', []),
                'old_final_messages': mt_result.get('final_messages', []),
                'old_judgment': mt_result.get('judgment', {}),
                'new_conversation': val_result.get('conversation', []),
                'new_final_messages': val_result.get('final_messages', []),
                'new_judgment': val_result.get('judgment', {})
            })
        elif mt_attack is True and val_attack is True:
            true_to_true.append(key)  # OLD success -> NEW success (maintained)
        elif mt_attack is False and val_attack is False:
            false_to_false.append(key)  # OLD failure -> NEW failure (maintained)
    
    # Print summary
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print(f"\nTotal matched entries: {len(common_keys)}")
    print(f"\nTransition Summary (OLD -> NEW):")
    print(f"  True -> True:   {len(true_to_true):5d} (attack still succeeds - BAD - still vulnerable)")
    print(f"  False -> False: {len(false_to_false):5d} (attack still blocked - GOOD - safety maintained)")
    print(f"  True -> False:  {len(true_to_false):5d} (attack now blocked - GOOD - security improved!)")
    print(f"  False -> True:  {len(false_to_true):5d} (attack now succeeds - BAD - new vulnerability!)")
    
    if missing_in_validation:
        print(f"\n  Missing attack_successful in validation: {len(missing_in_validation)}")
    if missing_in_multi_turn:
        print(f"  Missing attack_successful in multi_turn: {len(missing_in_multi_turn)}")
    
    # Detailed breakdown of regressions
    if true_to_false:
        print(f"\n{'='*80}")
        print(f"TRUE -> FALSE TRANSITIONS (SECURITY IMPROVEMENTS) - {len(true_to_false)} cases")
        print(f"OLD success -> NEW failure (these are GOOD - attacks that succeeded before are now blocked!)")
        print(f"{'='*80}")
        for i, case in enumerate(true_to_false[:20], 1):  # Show first 20
            print(f"\n{i}. Key: {case['key']}")
            print(f"   Question ID: {case['question_id']}")
            print(f"   Row Index: {case['row_idx']}")
            submission = case.get('submission_message', '')
            print(f"   Submission: {submission[:100]}{'...' if len(submission) > 100 else ''}")
        if len(true_to_false) > 20:
            print(f"\n... and {len(true_to_false) - 20} more cases")
    
    if false_to_true:
        print(f"\n{'='*80}")
        print(f"FALSE -> TRUE TRANSITIONS (NEW VULNERABILITIES) - {len(false_to_true)} cases")
        print(f"OLD failure -> NEW success (these are BAD - attacks that were blocked before now succeed!)")
        print(f"{'='*80}")
        for i, case in enumerate(false_to_true[:20], 1):  # Show first 20
            print(f"\n{i}. Key: {case['key']}")
            print(f"   Question ID: {case['question_id']}")
            print(f"   Row Index: {case['row_idx']}")
            submission = case.get('submission_message', '')
            print(f"   Submission: {submission[:100]}{'...' if len(submission) > 100 else ''}")
        if len(false_to_true) > 20:
            print(f"\n... and {len(false_to_true) - 20} more cases")
    
    # Calculate percentages
    total_transitions = len(true_to_false) + len(false_to_true) + len(true_to_true) + len(false_to_false)
    if total_transitions > 0:
        print(f"\n{'='*80}")
        print("PERCENTAGE BREAKDOWN")
        print(f"{'='*80}")
        print(f"  True -> True:   {len(true_to_true)/total_transitions*100:.2f}%")
        print(f"  False -> False: {len(false_to_false)/total_transitions*100:.2f}%")
        print(f"  True -> False:  {len(true_to_false)/total_transitions*100:.2f}% (REGRESSIONS)")
        print(f"  False -> True:  {len(false_to_true)/total_transitions*100:.2f}%")
    
    # Generate sklearn classification report
    if len(y_true) > 0 and len(y_pred) > 0:
        print(f"\n{'='*80}")
        print("SKLEARN CLASSIFICATION REPORT")
        print(f"{'='*80}")
        print("\nNote: multi_turn_results.json = OLD/BASELINE version")
        print("      validation_results.json = NEW/LoRA version")
        print("\nTreating OLD (multi_turn) as ground truth (y_true)")
        print("Treating NEW (validation) as predictions (y_pred)\n")
        
        # Classification report
        target_names = ['False (Attack Failed)', 'True (Attack Successful)']
        report = classification_report(
            y_true, y_pred,
            target_names=target_names,
            output_dict=False,
            zero_division=0
        )
        print(report)
        
        # Confusion matrix
        print(f"\n{'='*80}")
        print("CONFUSION MATRIX")
        print(f"{'='*80}")
        cm = confusion_matrix(y_true, y_pred, labels=[False, True])
        print("\n                Predicted")
        print("              False    True")
        print(f"Actual False  {cm[0,0]:5d}  {cm[0,1]:5d}")
        print(f"        True   {cm[1,0]:5d}  {cm[1,1]:5d}")
        print("\nInterpretation (OLD -> NEW) - All entries are attacks, False=blocked (GOOD), True=succeeds (BAD):")
        print(f"  False->False (TN): {cm[0,0]:5d} - OLD blocked -> NEW blocked (safety maintained - GOOD)")
        print(f"  False->True  (FP): {cm[0,1]:5d} - OLD blocked -> NEW succeeds (new vulnerability - BAD)")
        print(f"  True->False  (FN): {cm[1,0]:5d} - OLD succeeded -> NEW blocked (security improved - GOOD)")
        print(f"  True->True   (TP): {cm[1,1]:5d} - OLD succeeded -> NEW succeeds (still vulnerable - BAD)")
    
    # Save detailed results
    output_file = os.path.join(os.path.dirname(validation_file), "comparison_results.json")
    comparison_data = {
        'summary': {
            'total_matched': len(common_keys),
            'true_to_true': len(true_to_true),
            'false_to_false': len(false_to_false),
            'true_to_false': len(true_to_false),
            'false_to_true': len(false_to_true),
        },
        'true_to_false_cases': true_to_false,
        'false_to_true_cases': false_to_true,
    }
    
    # Add sklearn metrics if available (reuse cm from above if computed)
    if len(y_true) > 0 and len(y_pred) > 0:
        # Compute metrics for saving
        report_dict = classification_report(
            y_true, y_pred,
            target_names=['False (Attack Failed)', 'True (Attack Successful)'],
            output_dict=True,
            zero_division=0
        )
        cm_save = confusion_matrix(y_true, y_pred, labels=[False, True])
        comparison_data['sklearn_metrics'] = {
            'classification_report': report_dict,
            'confusion_matrix': {
                'matrix': cm_save.tolist(),
                'labels': ['False', 'True'],
                'interpretation': {
                    'true_negative': int(cm_save[0, 0]),
                    'false_positive': int(cm_save[0, 1]),
                    'false_negative': int(cm_save[1, 0]),
                    'true_positive': int(cm_save[1, 1])
                }
            }
        }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(comparison_data, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Detailed results saved to: {output_file}")
    print(f"{'='*80}\n")
    
    return comparison_data

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare attack_successful values between two result files")
    parser.add_argument(
        "--validation_file",
        type=str,
        default=None,
        help="Path to validation results JSON file"
    )
    parser.add_argument(
        "--multi_turn_file",
        type=str,
        default=None,
        help="Path to multi-turn results JSON file"
    )
    parser.add_argument(
        "--use_reevaluated",
        action="store_true",
        help="Use re-evaluated files instead of original files"
    )
    
    args = parser.parse_args()
    
    # Default paths
    script_dir = os.path.dirname(__file__)
    if args.validation_file is None:
        if args.use_reevaluated:
            validation_file = os.path.join(script_dir, "validation", "validation_results_reevaluated.json")
        else:
            validation_file = os.path.join(script_dir, "validation", "validation_results.json")
    else:
        validation_file = args.validation_file
    
    if args.multi_turn_file is None:
        if args.use_reevaluated:
            multi_turn_file = os.path.join(script_dir, "multi_turn_results_reevaluated.json")
        else:
            multi_turn_file = os.path.join(script_dir, "multi_turn_results.json")
    else:
        multi_turn_file = args.multi_turn_file
    
    if not os.path.exists(validation_file):
        print(f"Error: {validation_file} not found")
        return
    
    if not os.path.exists(multi_turn_file):
        print(f"Error: {multi_turn_file} not found")
        return
    
    compare_results(validation_file, multi_turn_file)

if __name__ == "__main__":
    main()

