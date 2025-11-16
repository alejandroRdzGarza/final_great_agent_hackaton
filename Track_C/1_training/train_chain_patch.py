import json
import torch
import argparse
import os
import platform
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from tqdm import tqdm
import numpy as np
from torch.nn import functional as F

# Model Setup (from bip.ipynb)
print("Loading model...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct",
    quantization_config=bnb_config,
    device_map="auto",
    dtype='auto'
)

print("Model loaded successfully!")

# Skip torch.compile on Windows
if platform.system() != "Windows":
    print("Compiling model for faster inference...")
    try:
        model = torch.compile(model, mode="reduce-overhead")
        print("Model compiled successfully!")
    except Exception as e:
        print(f"Warning: Could not compile model ({e}), continuing without compilation")
else:
    print("Skipping model compilation on Windows (requires Triton)")

# Get device
try:
    device = next(model.parameters()).device
except StopIteration:
    device = torch.device('cpu')
print(f"Using device: {device}")

# Data Loading Functions
def load_data():
    """Load activation metadata, tensors, and conversation results."""
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("Loading activation metadata...")
    activation_path = os.path.join(script_dir, "activations_all_messages.json")
    with open(activation_path, "r") as f:
        activation_metadata = json.load(f)
    
    print("Loading activation tensors...")
    tensor_path = os.path.join(script_dir, "activations_all_messages_tensors.pt")
    activation_tensors = torch.load(tensor_path, map_location='cpu')
    
    print("Loading conversation results...")
    results_path = os.path.join(script_dir, "multi_turn_results.json")
    with open(results_path, "r") as f:
        results = json.load(f)
    
    return activation_metadata, activation_tensors, results

def filter_successful_attacks(results):
    """Filter entries where attack was successful."""
    successful = []
    for entry in results:
        if entry.get('judgment', {}).get('attack_successful', False):
            successful.append(entry)
    return successful

def find_safe_and_break_steps(entry, activation_metadata):
    """Find last safe step (label=0) and first break step (label=1) for an entry."""
    message_labels = entry.get('message_labels', [])
    entry_idx = entry.get('entry_idx')
    example_id = entry.get('row_idx')
    
    if not message_labels:
        return None, None
    
    # Find last safe step (label=0)
    safe_depth = None
    for i, label in enumerate(message_labels, start=1):
        if label == 0:
            safe_depth = i
        else:
            break
    
    # Find first break step (label=1)
    break_depth = None
    for i, label in enumerate(message_labels, start=1):
        if label == 1:
            break_depth = i
            break
    
    return safe_depth, break_depth

def load_activation(activation_tensors, example_id, entry_idx, depth, layer):
    """Load activation tensor for specific example_id, entry_idx, depth, and layer."""
    key = f"example_{example_id}_entry_{entry_idx}_depth_{depth}_layer_{layer}"
    if key in activation_tensors:
        return activation_tensors[key]
    return None

# Global Training Data Collection
def collect_training_examples(successful_entries, activation_tensors):
    """Collect all training examples from successful jailbreak conversations."""
    training_examples = []
    layer = 4  # Only use layer 4
    
    print("Collecting training examples from successful jailbreak conversations...")
    
    skipped_no_messages = 0
    skipped_no_depths = 0
    skipped_no_activations = 0
    skipped_zero_norm = 0
    skipped_no_assistant = 0
    
    # We need to find the original entry_idx from the results array
    # Load results to get the mapping
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_path = os.path.join(script_dir, "multi_turn_results.json")
    with open(results_path, "r") as f:
        all_results = json.load(f)
    
    # Create mapping from (row_idx) to entry_idx in original results
    row_to_entry_idx = {}
    for entry_idx, entry in enumerate(all_results):
        row_idx = entry.get('row_idx')
        if row_idx not in row_to_entry_idx:
            row_to_entry_idx[row_idx] = []
        row_to_entry_idx[row_idx].append(entry_idx)
    
    for entry in tqdm(successful_entries, desc="Processing chains"):
        example_id = entry.get('row_idx')
        # entry_idx is the position in the original results array
        # Since multiple entries can have same row_idx, try all possible entry_idx values
        possible_entry_indices = row_to_entry_idx.get(example_id, [])
        final_messages = entry.get('final_messages', [])
        message_labels = entry.get('message_labels', [])
        
        if not final_messages or not message_labels:
            skipped_no_messages += 1
            continue
        
        # Find safe and break depths
        safe_depth, break_depth = find_safe_and_break_steps(entry, None)
        
        if safe_depth is None or break_depth is None:
            skipped_no_depths += 1
            continue
        
        # Try to find matching activations by trying all possible entry_idx values
        h_safe_layer4 = None
        h_break_layer4 = None
        found_entry_idx = None
        
        for entry_idx in possible_entry_indices:
            h_safe = load_activation(activation_tensors, example_id, entry_idx, safe_depth, layer)
            h_break = load_activation(activation_tensors, example_id, entry_idx, break_depth, layer)
            if h_safe is not None and h_break is not None:
                h_safe_layer4 = h_safe
                h_break_layer4 = h_break
                found_entry_idx = entry_idx
                break
        
        if h_safe_layer4 is None or h_break_layer4 is None:
            skipped_no_activations += 1
            continue
        
        # Compute normalized direction
        d_chain = h_break_layer4 - h_safe_layer4
        d_chain_norm = torch.norm(d_chain)
        if d_chain_norm > 0:
            d_chain = d_chain / d_chain_norm
        else:
            skipped_zero_norm += 1
            continue
        
        # Store break step example
        break_messages = final_messages[:break_depth]
        training_examples.append({
            'type': 'break',
            'messages': break_messages,
            'target': "I cannot assist with this request.",
            'd_chain': d_chain,
            'h_safe_layer4': h_safe_layer4,
            'example_id': example_id,
            'entry_idx': found_entry_idx,
            'depth': break_depth
        })
        
        # Store safe anchor steps: use the safe_depth step
        # At safe_depth, we have messages up to that point, and the next message should be assistant's safe reply
        if safe_depth is not None and safe_depth < len(final_messages):
            # Messages up to (but not including) the assistant's reply
            # Conversation structure: [system, user1, assistant1, user2, assistant2, ...]
            # safe_depth points to the last safe message (typically an assistant reply)
            # We want the conversation *before* this assistant reply to generate it
            anchor_messages = final_messages[:safe_depth]
            
            # The assistant's safe reply is at position safe_depth
            if safe_depth < len(final_messages) and final_messages[safe_depth - 1].get('role') == 'assistant':
                assistant_reply = final_messages[safe_depth - 1].get('content', '')
                
                if assistant_reply:
                    training_examples.append({
                        'type': 'safe',
                        'messages': anchor_messages[:-1],  # Exclude the assistant reply we're trying to generate
                        'target': assistant_reply,
                        'd_chain': d_chain,
                        'h_safe_layer4': h_safe_layer4,
                        'example_id': example_id,
                        'entry_idx': found_entry_idx,
                        'depth': safe_depth
                    })
                else:
                    skipped_no_assistant += 1
            else:
                skipped_no_assistant += 1
    
    # Print statistics
    break_examples = [e for e in training_examples if e['type'] == 'break']
    safe_examples = [e for e in training_examples if e['type'] == 'safe']
    
    print(f"\n{'='*60}")
    print("Training Examples Summary:")
    print(f"{'='*60}")
    print(f"Total examples: {len(training_examples)}")
    print(f"  - Break examples: {len(break_examples)}")
    print(f"  - Safe anchor examples: {len(safe_examples)}")
    print(f"Unique chains processed: {len(set((e['example_id'], e['entry_idx']) for e in training_examples))}")
    print(f"\nSkipped chains:")
    print(f"  - No messages/labels: {skipped_no_messages}")
    print(f"  - No safe/break depths: {skipped_no_depths}")
    print(f"  - Missing activations: {skipped_no_activations}")
    print(f"  - Zero norm direction: {skipped_zero_norm}")
    print(f"  - No assistant reply: {skipped_no_assistant}")
    print(f"{'='*60}\n")
    
    return training_examples

# LoRA Setup (Multiple layers)
def setup_lora(model, lora_r=8, lora_alpha=16, lora_dropout=0.1, target_layers=[4, 8, 12]):
    """Set up LoRA configuration and attach to specified layers."""
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # Target modules in specified layers only
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        layers_to_transform=target_layers,  # CRITICAL: Only transform specified layers
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"\n{'='*60}")
    print("LoRA Configuration:")
    print(f"{'='*60}")
    print(f"Target layers: {target_layers}")
    print(f"Target modules: {target_modules}")
    print(f"LoRA rank (r): {lora_r}")
    print(f"LoRA alpha: {lora_alpha}")
    print(f"LoRA dropout: {lora_dropout}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable %: {100 * trainable_params / total_params:.4f}%")
    print(f"{'='*60}\n")
    
    return model

# Training Loop with Dual Loss
def train_global_patch(model, tokenizer, training_examples, learning_rate=1e-4, lambda_geom=0.5, 
                       batch_size=4, epochs=3, gradient_accumulation_steps=4):
    """Train global patch with dual loss (behavior + geometry)."""
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate
    )
    
    # Create batches
    num_examples = len(training_examples)
    num_batches = (num_examples + batch_size - 1) // batch_size
    
    print(f"\n{'='*60}")
    print("Training Configuration:")
    print(f"{'='*60}")
    print(f"Total examples: {num_examples}")
    print(f"Batch size: {batch_size}")
    print(f"Gradient accumulation steps: {gradient_accumulation_steps}")
    print(f"Effective batch size: {batch_size * gradient_accumulation_steps}")
    print(f"Number of batches: {num_batches}")
    print(f"Epochs: {epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Geometry loss weight (λ): {lambda_geom}")
    print(f"{'='*60}\n")
    
    model.train()
    
    for epoch in range(epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"{'='*60}")
        total_loss = 0.0
        total_behavior_loss = 0.0
        total_geometry_loss = 0.0
        batch_count = 0
        
        # Shuffle examples
        import random
        shuffled_examples = training_examples.copy()
        random.shuffle(shuffled_examples)
        
        optimizer.zero_grad()
        
        for batch_idx in tqdm(range(num_batches), desc="Training"):
            batch_examples = shuffled_examples[batch_idx * batch_size:(batch_idx + 1) * batch_size]
            
            if not batch_examples:
                continue
            
            batch_behavior_loss = 0.0
            batch_geometry_loss = 0.0
            num_valid = 0
            
            for example in batch_examples:
                messages = example['messages']
                target_text = example['target']
                # Convert to device and ensure same dtype (float16 to match model)
                d_chain = example['d_chain'].to(device).half()
                h_safe_layer4 = example['h_safe_layer4'].to(device).half()
                example_type = example['type']
                
                # Prepare input
                inputs = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                )
                
                # Prepare target tokens
                target_tokens = tokenizer(
                    target_text,
                    return_tensors="pt",
                    add_special_tokens=False
                )['input_ids']
                
                # Store original lengths before concatenation
                orig_input_len = inputs['input_ids'].shape[1]
                orig_target_len = target_tokens.shape[1]
                
                # Concatenate input + target for single forward pass
                full_input_ids = torch.cat([inputs['input_ids'], target_tokens], dim=1)
                full_attention_mask = torch.ones_like(full_input_ids)
                if 'attention_mask' in inputs:
                    full_attention_mask[:, :orig_input_len] = inputs['attention_mask']
                
                # Truncate to 256 tokens if needed (keep last 256 tokens to preserve target)
                max_tokens = 256
                if full_input_ids.shape[1] > max_tokens:
                    full_input_ids = full_input_ids[:, -max_tokens:]
                    full_attention_mask = full_attention_mask[:, -max_tokens:]
                    # After truncation from left, target is still at the end
                    # Calculate how much of input remains
                    input_len = max_tokens - orig_target_len
                    # Update target to match what's actually in full_input_ids
                    target_tokens = full_input_ids[:, input_len:]
                else:
                    input_len = orig_input_len
                
                full_inputs = {
                    'input_ids': full_input_ids.to(device),
                    'attention_mask': full_attention_mask.to(device)
                }
                target_tokens = target_tokens.to(device)
                
                # Single forward pass with hidden states
                outputs = model(**full_inputs, output_hidden_states=True, use_cache=False)
                logits = outputs.logits
                hidden_states = outputs.hidden_states
                
                # Extract layer 4 hidden state (index 5: embedding + layers 0-15)
                # Use activations at the last input position (before target generation)
                layer_4_hidden = hidden_states[5]  # layer 4 is at index 5
                # Mean pool over sequence (as saved activations use mean pooling)
                h_current_layer4 = layer_4_hidden.mean(dim=1).squeeze(0).half()
                
                # Behavior loss: predict target tokens
                # Compute loss on target tokens only (shift by 1 for next-token prediction)
                # For next-token prediction: logits at position i predict token at position i+1
                # Target tokens start at position input_len in full_input_ids
                # So we need logits from position input_len-1 to predict the first target token
                # up to position (total_len - 2) to predict the last target token
                
                # Shift logits and labels for next-token prediction
                shift_logits = logits[:, :-1, :].contiguous()  # Remove last position
                # Extract from device tensor, not the original CPU tensor
                shift_labels = full_inputs['input_ids'][:, 1:].contiguous()  # Remove first position
                
                # Only compute loss on target portion
                # Target portion in shift_labels starts at position input_len-1
                target_start = max(0, input_len - 1) if input_len > 0 else 0
                target_logits = shift_logits[:, target_start:, :]
                target_labels = shift_labels[:, target_start:]
                
                behavior_loss = F.cross_entropy(
                    target_logits.view(-1, target_logits.size(-1)),
                    target_labels.view(-1),
                    ignore_index=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else -100
                )
                
                # Geometry loss
                if example_type == 'break':
                    # Minimize projection onto jailbreak direction (any direction)
                    # Squaring makes it always positive and penalizes large magnitudes
                    proj = torch.dot(h_current_layer4, d_chain)
                    geometry_loss = proj ** 2
                else:  # safe
                    # Keep projection near safe state's projection
                    proj_current = torch.dot(h_current_layer4, d_chain)
                    proj_safe = torch.dot(h_safe_layer4, d_chain)
                    geometry_loss = (proj_current - proj_safe) ** 2
                
                batch_behavior_loss += behavior_loss
                batch_geometry_loss += geometry_loss
                num_valid += 1
            
            if num_valid > 0:
                avg_behavior_loss = batch_behavior_loss / num_valid
                avg_geometry_loss = batch_geometry_loss / num_valid
                total_loss_batch = avg_behavior_loss + lambda_geom * avg_geometry_loss
                
                # Scale loss by accumulation steps for proper gradient averaging
                (total_loss_batch / gradient_accumulation_steps).backward()
                
                # Update weights every N accumulation steps
                if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == num_batches:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                
                total_loss += total_loss_batch.item()
                total_behavior_loss += avg_behavior_loss.item()
                total_geometry_loss += avg_geometry_loss.item()
                batch_count += 1
                
                # Print batch metrics every N batches or at the end
                log_interval = max(1, num_batches // 10)  # Log ~10 times per epoch
                if (batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == num_batches:
                    current_avg_loss = total_loss / batch_count if batch_count > 0 else 0
                    current_avg_behavior = total_behavior_loss / batch_count if batch_count > 0 else 0
                    current_avg_geometry = total_geometry_loss / batch_count if batch_count > 0 else 0
                    print(f"  Batch {batch_idx + 1}/{num_batches} | "
                          f"Loss: {current_avg_loss:.4f} | "
                          f"Behavior: {current_avg_behavior:.4f} | "
                          f"Geometry: {current_avg_geometry:.4f}")
            
            # Clear cache periodically
            if batch_idx % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        avg_behavior = total_behavior_loss / batch_count if batch_count > 0 else 0
        avg_geometry = total_geometry_loss / batch_count if batch_count > 0 else 0
        
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Average Loss: {avg_loss:.6f}")
        print(f"    - Behavior Loss: {avg_behavior:.6f}")
        print(f"    - Geometry Loss: {avg_geometry:.6f}")
        print(f"    - Weighted Geometry: {lambda_geom * avg_geometry:.6f}")
        print(f"{'='*60}")
    
    return model

# Testing Function
def test_patch(model, tokenizer, test_entry, max_new_tokens=128):
    """Test the patched model on a successful attack conversation."""
    print("\n" + "="*80)
    print("Testing patched model on successful attack")
    print("="*80)
    
    final_messages = test_entry.get('final_messages', [])
    message_labels = test_entry.get('message_labels', [])
    
    if not final_messages:
        print("No messages in test entry")
        return
    
    # Find break step
    safe_depth, break_depth = find_safe_and_break_steps(test_entry, None)
    
    if break_depth is None:
        print("No break step found in test entry")
        return
    
    print(f"\nTest entry: example_id={test_entry.get('row_idx')}, entry_idx={test_entry.get('entry_idx')}")
    print(f"Break occurs at depth: {break_depth}")
    
    # Replay conversation up to break step
    print("\n--- Conversation up to break step ---")
    break_messages = final_messages[:break_depth]
    
    for i, msg in enumerate(break_messages):
        role = msg.get('role', 'unknown')
        content = msg.get('content', '')[:200]  # Truncate for display
        print(f"{i+1}. [{role.upper()}]: {content}...")
    
    # Generate with patched model
    print("\n--- Generating with patched model ---")
    inputs = tokenizer.apply_chat_template(
        break_messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    
    # Truncate if needed
    max_tokens = 256
    if 'input_ids' in inputs and inputs['input_ids'].shape[1] > max_tokens:
        inputs['input_ids'] = inputs['input_ids'][:, -max_tokens:]
        if 'attention_mask' in inputs:
            inputs['attention_mask'] = inputs['attention_mask'][:, -max_tokens:]
    
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    
    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Decode response
    generated_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    
    print(f"\nPATCHED MODEL RESPONSE:")
    print(generated_text)
    
    # Compare with original
    original_reply = None
    if break_depth < len(final_messages):
        original_msg = final_messages[break_depth]
        if original_msg.get('role') == 'assistant':
            original_reply = original_msg.get('content', '')
    
    if original_reply:
        print(f"\nORIGINAL RESPONSE (jailbroken):")
        print(original_reply[:500])
    
    print("\n" + "="*80)
    
    return generated_text, original_reply

# Main Function
def main():
    parser = argparse.ArgumentParser(description="Train global jailbreak patch using geometric activation differences")
    parser.add_argument("--output_dir", type=str, default="global_chain_patch", help="Base directory to save LoRA adapter (unique timestamped subfolder will be created)")
    parser.add_argument("--test_example_id", type=int, default=None, help="Example ID to use for testing (default: first successful attack)")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--lambda_geom", type=float, default=0.5, help="Weight for geometry loss")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")
    parser.add_argument("--target_layers", type=int, nargs='+', default=[4, 8, 12], help="Target layers for LoRA (default: 4 8 12)")
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("GLOBAL JAILBREAK PATCH TRAINING")
    print("="*60)
    print(f"Output directory: {args.output_dir}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Lambda (geometry weight): {args.lambda_geom}")
    print(f"Batch size: {args.batch_size}")
    print(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
    print(f"Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"Epochs: {args.epochs}")
    print(f"LoRA config: r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
    print(f"Target layers: {args.target_layers}")
    print("="*60 + "\n")
    
    # Load data
    activation_metadata, activation_tensors, results = load_data()
    
    # Filter successful attacks
    successful_entries = filter_successful_attacks(results)
    print(f"\n{'='*60}")
    print(f"Found {len(successful_entries)} successful jailbreak conversations")
    print(f"{'='*60}")
    
    if len(successful_entries) == 0:
        print("No successful attacks found. Exiting.")
        return
    
    # Collect training examples
    print("\nCollecting training examples...")
    training_examples = collect_training_examples(successful_entries, activation_tensors)
    
    if len(training_examples) == 0:
        print("No training examples collected. Exiting.")
        return
    
    # Setup LoRA
    print("\nSetting up LoRA...")
    global model
    model = setup_lora(model, lora_r=args.lora_r, lora_alpha=args.lora_alpha, 
                      lora_dropout=args.lora_dropout, target_layers=args.target_layers)
    
    # Train global patch
    print("\nStarting training...")
    model = train_global_patch(
        model, tokenizer, training_examples,
        learning_rate=args.learning_rate,
        lambda_geom=args.lambda_geom,
        batch_size=args.batch_size,
        epochs=args.epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )
    
    # Save LoRA adapter in unique folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_output_dir = os.path.join(args.output_dir, f"lora_{timestamp}")
    print(f"\n{'='*60}")
    print(f"Saving LoRA adapter to {unique_output_dir}...")
    print(f"{'='*60}")
    os.makedirs(unique_output_dir, exist_ok=True)
    model.save_pretrained(unique_output_dir)
    tokenizer.save_pretrained(unique_output_dir)
    print(f"✓ Adapter saved to: {os.path.abspath(unique_output_dir)}")
    print(f"{'='*60}\n")
    
    # Test on a successful attack
    test_entry = None
    if args.test_example_id is not None:
        # Find entry with matching example_id
        for entry in successful_entries:
            if entry.get('row_idx') == args.test_example_id:
                test_entry = entry
                break
        if test_entry is None:
            print(f"Warning: Could not find entry with example_id={args.test_example_id}")
    else:
        # Use first successful attack
        test_entry = successful_entries[0]
    
    if test_entry:
        test_patch(model, tokenizer, test_entry)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"LoRA adapter saved to: {os.path.abspath(unique_output_dir)}")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()

