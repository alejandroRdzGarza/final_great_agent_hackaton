import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import os
import platform
from tqdm import tqdm

# Load model like in bip.ipynb
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

# Compile model for faster inference (skip on Windows as it requires Triton)
# Note: torch.compile requires Triton which is not well-supported on Windows
if platform.system() != "Windows":
    print("Compiling model for faster inference...")
    try:
        model = torch.compile(model, mode="reduce-overhead")
        print("Model compiled successfully!")
    except Exception as e:
        print(f"Warning: Could not compile model ({e}), continuing without compilation")
else:
    print("Skipping model compilation on Windows (requires Triton)")

# Verify model structure
if not (hasattr(model, 'model') and hasattr(model.model, 'layers')):
    print("Could not find model.model.layers structure")
    exit(1)

num_layers = len(model.model.layers)
print(f"Model has {num_layers} layers")

# Target layers for activation extraction
target_layers = [4, 8, 12]
for layer_idx in target_layers:
    if layer_idx >= num_layers:
        print(f"Warning: Layer {layer_idx} does not exist (model has {num_layers} layers)")
        exit(1)

print(f"Target layers: {target_layers}")

# Load multi_turn_results.json
print("Loading results...")
with open("multi_turn_results.json", "r") as f:
    results = json.load(f)

print(f"Loaded {len(results)} entries")

# Get device (for quantized models with device_map="auto")
try:
    device = next(model.parameters()).device
except StopIteration:
    device = torch.device('cpu')
print(f"Using device: {device}")

# Process all entries
print("\nProcessing all entries...")
all_activations = []

for entry_idx, entry in enumerate(tqdm(results, desc="Processing entries")):
    final_messages = entry.get('final_messages', [])
    
    if not final_messages:
        continue
    
    row_idx = entry.get('row_idx')
    question_id = entry.get('question_id')
    message_labels = entry.get('message_labels', [])
    turn_count = entry.get('turn_count', 0)
    
    # Process each message step (taking cumulative history up to that step)
    # Step 1: [user0] -> hidden state
    # Step 2: [user0, assistant0] -> hidden state
    # Step 3: [user0, assistant0, user1] -> hidden state
    # etc. Each step truncates to latest 256 tokens if needed
    for step in tqdm(range(1, len(final_messages) + 1), desc=f"  Entry {entry_idx} steps", leave=False):
        # Get cumulative messages up to this step
        cumulative_messages = final_messages[:step]
        
        # Get message label for this step (step is 1-indexed, array is 0-indexed)
        # message_labels corresponds to each message in final_messages
        message_label = None
        if message_labels and step <= len(message_labels):
            message_label = message_labels[step - 1]
        
        # Prepare input from cumulative messages
        inputs = tokenizer.apply_chat_template(
            cumulative_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        
        # Truncate to keep only the latest 256 tokens of the cumulative conversation
        # This ensures consistent processing time regardless of conversation length
        max_tokens = 256
        if 'input_ids' in inputs and inputs['input_ids'].shape[1] > max_tokens:
            inputs['input_ids'] = inputs['input_ids'][:, -max_tokens:]
            if 'attention_mask' in inputs:
                inputs['attention_mask'] = inputs['attention_mask'][:, -max_tokens:]
        
        # Move inputs to device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        # Run forward pass with output_hidden_states=True
        # use_cache=False since we don't need past key values for forward-only passes
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True, use_cache=False)
        
        # Extract hidden states for target layers
        # hidden_states is a tuple: (embedding_output, layer_0, layer_1, ..., layer_n)
        # So layer i is at index i+1
        hidden_states = outputs.hidden_states
        
        # Store each layer activation with full metadata
        # Process all layers first, then move to CPU in batch for better efficiency
        step_activations = {}
        for layer_idx in target_layers:
            # Get hidden states for this layer (index is layer_idx + 1)
            layer_hidden = hidden_states[layer_idx + 1]
            
            # Compute global mean embedding over sequence dimension (dim=1)
            # Shape: (batch_size, seq_len, hidden_size) -> (batch_size, hidden_size)
            global_mean = layer_hidden.mean(dim=1)
            
            # If batch_size is 1, squeeze to get (hidden_size,)
            if global_mean.shape[0] == 1:
                global_mean = global_mean.squeeze(0)
            
            step_activations[layer_idx] = global_mean
        
        # Explicitly delete large tensors to free GPU memory immediately
        del outputs, hidden_states, inputs
        
        # Move all activations to CPU at once (more efficient than individual transfers)
        for layer_idx, global_mean in step_activations.items():
            global_mean_cpu = global_mean.detach().cpu()
            
            # Store activation with full metadata
            all_activations.append({
                'example_id': row_idx,  # Using row_idx as example_id
                'question_id': question_id,
                'entry_idx': entry_idx,
                'depth': step,  # Conversation depth (number of messages in history)
                'model_layer': layer_idx,  # Model layer (4, 8, or 12)
                'message_label': message_label,  # Label from message_labels array
                'turn_count': turn_count,  # Total turns in conversation
                'activation': global_mean_cpu,
                'activation_shape': list(global_mean_cpu.shape)
            })
        
        # Clear step activations
        del step_activations
        
        # Periodically clear CUDA cache to prevent memory fragmentation
        # Do this every 10 steps to avoid overhead
        if step % 10 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

# Save all activations
output_file = "activations_all_messages.json"
print(f"\nSaving activations metadata to {output_file}...")

# Save metadata (without tensors, just shapes and metadata)
activation_data = {
    'metadata': {
        'target_layers': target_layers,
        'model_name': 'meta-llama/Llama-3.2-1B-Instruct',
        'total_entries': len(results),
        'total_activations': len(all_activations)
    },
    'activations': []
}

for act_data in all_activations:
    activation_data['activations'].append({
        'example_id': act_data['example_id'],
        'question_id': act_data['question_id'],
        'entry_idx': act_data['entry_idx'],
        'depth': act_data['depth'],
        'model_layer': act_data['model_layer'],
        'message_label': act_data['message_label'],
        'turn_count': act_data['turn_count'],
        'activation_shape': act_data['activation_shape']
    })

with open(output_file, 'w') as f:
    json.dump(activation_data, f, indent=2)

# Save actual tensors separately
tensor_file = "activations_all_messages_tensors.pt"
print(f"Saving tensor data to {tensor_file}...")
tensor_data = {}
for idx, act_data in enumerate(all_activations):
    # Create a unique key with all identifying information
    key = f"example_{act_data['example_id']}_entry_{act_data['entry_idx']}_depth_{act_data['depth']}_layer_{act_data['model_layer']}"
    tensor_data[key] = act_data['activation']

torch.save(tensor_data, tensor_file)

print(f"\nDone! Saved:")
print(f"  - Metadata: {output_file}")
print(f"  - Tensors: {tensor_file}")
print(f"  - Total entries processed: {len(results)}")
print(f"  - Total activations: {len(all_activations)}")
print(f"  - Layers extracted: {target_layers}")
print(f"  - Each activation includes: example_id, depth, model_layer, message_label")

