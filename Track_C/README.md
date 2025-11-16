# G-BRACE: Geometric Boundary Repair for Adversarial Chain Evasion

A defense mechanism for language models that uses geometric activation analysis to patch jailbreak vulnerabilities. This project trains a LoRA adapter that reinforces safe behavior by learning from successful jailbreak attempts.

## Overview

G-BRACE (Geometric Boundary Repair for Adversarial Chain Evasion) is a training pipeline that:

1. **Analyzes jailbreak attacks** - Identifies the geometric activation differences between safe and compromised model states
2. **Trains a defensive patch** - Uses LoRA to learn a global patch that prevents jailbreak attempts
3. **Validates effectiveness** - Evaluates the patched model on MMLU and refusal rate metrics

The method works by:
- Extracting hidden state activations from successful jailbreak conversations
- Computing the geometric direction vector between "safe" and "compromised" states
- Training a LoRA adapter to minimize projection onto jailbreak directions while maintaining task performance

## Project Structure

```
Doctor/
├── 1_training/              # Training scripts
│   └── train_chain_patch.py # Main training script for G-BRACE patch
├── 2_evaluation/            # Evaluation scripts
│   ├── evaluate_mmlu.py    # MMLU benchmark evaluation
│   ├── multi_turn_eval.py   # Multi-turn conversation evaluation
│   └── multi_turn_eval_8b.py
├── 3_data_processing/       # Data preparation and processing
│   ├── label_messages.py   # Label conversation messages (safe/jailbreak)
│   ├── save_activations.py # Extract and save model activations
│   ├── compare_results.py  # Compare evaluation results
│   └── reevaluate_attack_successful.py
├── 4_data/                  # Data files (activations, results)
│   ├── activations_all_messages_tensors.pt
│   ├── activations_all_messages.json
│   └── multi_turn_results.json
├── 5_models/                # Trained LoRA adapters
│   └── global_chain_patch/
├── 6_validation/            # Validation scripts and results
│   └── validation/
│       └── validate_lora.py
├── 7_notebooks/             # Jupyter notebooks for analysis
├── 8_generalize/            # Generalization tests
│   ├── multi_turn_eval_lora.py
│   └── multi_turn_eval_openrouter.py
└── 9_visualisation/         # Visualization scripts
    └── llama_results_chart.py
```

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended)
- Hugging Face account with access to:
  - `meta-llama/Llama-3.2-1B-Instruct` (base model)
  - `ScaleAI/mhj` (MultiTurn Human Jailbreak dataset)
- API keys for evaluation (OpenAI, OpenRouter) - set as environment variables

Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset

This project uses the **HarmBench** dataset from ScaleAI for training and evaluation:
- **Dataset**: `ScaleAI/mhj/harmbench_behaviors.csv`
- **Description**: A collection of harmful behaviors and multi-turn jailbreak attempts
- **Usage**: First 200 examples are used for training and evaluation
- **Reference**: [HarmBench: A Standardized Evaluation Framework for Backdoor and Alignment Robustness](https://github.com/centerforaisafety/HarmBench)

For task performance evaluation, we also use:
- **MMLU** (Massive Multitask Language Understanding): `cais/mmlu`

## Usage

### 1. Data Collection

First, run multi-turn conversations to collect jailbreak attempts:

```bash
cd 2_evaluation
python multi_turn_eval.py
```

This generates `multi_turn_results.json` with conversation logs and attack success labels.

### 2. Label Messages

Label conversation messages to identify where jailbreaks occur:

```bash
cd 3_data_processing
python label_messages.py
```

This uses an LLM judge to identify the exact message where each jailbreak succeeded.

### 3. Extract Activations

Extract hidden state activations from conversations:

```bash
cd 3_data_processing
python save_activations.py
```

This generates:
- `activations_all_messages.json` - Metadata about activations
- `activations_all_messages_tensors.pt` - Activation tensors

### 4. Train G-BRACE Patch

Train the LoRA adapter using geometric activation differences:

```bash
cd 1_training
python train_chain_patch.py \
    --output_dir ../5_models/global_chain_patch \
    --learning_rate 1e-4 \
    --lambda_geom 0.5 \
    --batch_size 4 \
    --epochs 3 \
    --target_layers 4 8 12
```

**Key Parameters:**
- `--lambda_geom`: Weight for geometry loss (controls trade-off between behavior and geometry)
- `--target_layers`: Transformer layers to apply LoRA (default: 4, 8, 12)
- `--lora_r`: LoRA rank (default: 8)
- `--lora_alpha`: LoRA alpha scaling (default: 16)

The training uses a dual loss:
- **Behavior loss**: Standard language modeling loss for target responses
- **Geometry loss**: Minimizes projection onto jailbreak direction vectors

### 5. Evaluate on MMLU

Evaluate task performance on MMLU benchmark:

```bash
cd 2_evaluation
python evaluate_mmlu.py \
    --lora_path ../5_models/global_chain_patch/lora_YYYYMMDD_HHMMSS \
    --tasks mmlu
```

### 6. Validate on Test Set

Validate the patched model on a held-out test set:

```bash
cd 6_validation/validation
python validate_lora.py
```

This measures refusal rates on validation and training splits.

### 7. Generalization Tests

Test the patch on different models/APIs:

```bash
cd 8_generalize
python multi_turn_eval_lora.py      # Local LoRA model
python multi_turn_eval_openrouter.py # OpenRouter API
```

### 8. Visualize Results

Generate comparison charts:

```bash
cd 9_visualisation
python llama_results_chart.py
```

## How It Works

### Geometric Activation Analysis

1. **Safe State**: Last message before jailbreak (label=0)
2. **Break State**: First message where jailbreak occurs (label=1)
3. **Direction Vector**: `d_chain = normalize(h_break - h_safe)`

### Training Objective

For each training example:
- **Break examples**: Minimize `(h_current · d_chain)²` to prevent jailbreak
- **Safe examples**: Keep `(h_current · d_chain)` near `(h_safe · d_chain)` to maintain safe behavior

Combined loss:
```
L = L_behavior + λ_geom * L_geometry
```

### LoRA Configuration

- **Target layers**: 4, 8, 12 (middle layers)
- **Target modules**: `q_proj`, `k_proj`, `v_proj`, `o_proj`
- **Rank**: 8 (low-rank adaptation)
- **Alpha**: 16 (scaling factor)

## Results

The patched model shows:
- **MMLU**: Maintains task performance (~42-43%)
- **Refusal Rate (Validation)**: Increases from ~61% to ~88%
- **Refusal Rate (Train)**: Increases from ~68% to ~95%

See `9_visualisation/llama_results_chart.svg` for visual comparison.

## Data Format

### `multi_turn_results.json`
```json
{
  "row_idx": 0,
  "question_id": "...",
  "final_messages": [...],
  "message_labels": [0, 0, 0, 1, 1, ...],
  "judgment": {
    "attack_successful": true,
    "score": 8
  }
}
```

### Activation Files
- **JSON**: Metadata mapping `(example_id, entry_idx, depth, layer)` to tensor keys
- **PT**: PyTorch tensors with mean-pooled hidden states from layer 4

## Notes

- The project uses 4-bit quantization (BitsAndBytes) for memory efficiency
- Windows users: `torch.compile` is skipped (requires Triton)
- API keys should be set as environment variables:
  ```bash
  export OPENAI_API_KEY="your-openai-key"
  export OPENROUTER_API_KEY="your-openrouter-key"
  ```
  Or on Windows:
  ```powershell
  $env:OPENAI_API_KEY="your-openai-key"
  $env:OPENROUTER_API_KEY="your-openrouter-key"
  ```
- Visualization charts can be regenerated by running `9_visualisation/llama_results_chart.py`
- Data files and model checkpoints are excluded from git (see `.gitignore`)

## License

MIT License

Copyright (c) 2024

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Citation

If you use this code, please cite:
```
CRD2-GreatAgentHack2025- Nichita Mitrea, Stefan Mandris, Tom Houghton, Jesus Garza, Pable Islas.
```

