# Copolymer Block Distribution Generation via Conditional Diffusion

A deep learning framework for generating polymer block length distributions using conditional diffusion models with Set Transformer-based condition encoding and Mayo-Lewis theoretical analysis.

## 🔬 Overview

This project implements a state-of-the-art approach to predict and generate polymer block length distributions from chemical composition and polymerization conditions. The system combines:

- **Set Transformer Architecture**: Handles variable-length polymer chain sequences efficiently
- **Conditional Diffusion Model**: DiT-1D (Diffusion Transformer) for histogram generation  
- **Mayo-Lewis Theory Integration**: Theoretical analysis for validation and comparison
- **v-Parameterization**: Stable diffusion training with cosine noise scheduling
- **Classifier-Free Guidance**: Enhanced generation quality during inference

## 📁 Project Structure

```
copolymer/
├── src/                          # Core modules
│   ├── encoder.py               # Set Transformer condition encoder
│   ├── diffusion.py             # DiT-1D diffusion model & DDIM sampler
│   ├── decoder.py               # Histogram decoding utilities
│   └── dataset.py               # Data loading and preprocessing
├── data/                        # Data processing
│   ├── dataset.py               # ChainSetDataset implementation
│   ├── block_dist.py            # Feature extraction & Mayo-Lewis theory
│   └── copolymer.csv            # Polymer dataset
├── condition_encoder_trainer.py # Train condition encoder
├── diffusion_trainer.py         # Train diffusion model
├── diffusion_inference.py       # Inference and visualization
├── prediction_eval.py           # Comprehensive evaluation pipeline
├── requirements.txt             # Dependencies
└── README.md                    # This file
```

## 🚀 Features

### Core Capabilities
- **Multi-Modal Input**: Processes polymer sequences, chemical composition, and reaction conditions
- **Theoretical Validation**: Integrates Mayo-Lewis copolymerization theory for comparison
- **Real-Time Monitoring**: In-training evaluation with automatic best model selection
- **Flexible Architecture**: Supports different bin sizes, guidance scales, and temperature settings

### Advanced Training Features
- **Fixed Test Set Evaluation**: Consistent evaluation on specified test indices
- **Epoch-wise Visualization**: Generates top-k visualizations during training
- **Best Model Selection**: Automatically saves model with lowest test KL divergence
- **Inference-Only Mode**: Skip training and directly evaluate pre-trained models

### Visualization & Analysis
- **Mayo-Lewis Overlays**: Red dashed theoretical curves on all plots
- **Top-K Grid Generation**: Comprehensive comparison grids (ground truth vs prediction vs theory)
- **Detailed Rankings**: CSV exports with KL divergence and EMD metrics
- **Professional Styling**: Consistent color schemes and typography

## 📋 Requirements

```bash
# Core ML and Scientific Computing
torch>=1.9.0
numpy>=1.21.0
pandas>=1.3.0

# Visualization and Plotting
matplotlib>=3.4.0

# Machine Learning and Analysis
scikit-learn>=1.0.0
scipy>=1.7.0  # Required for pearson correlation in block_dist.py

# Progress bars and utilities
tqdm>=4.62.0
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## 🔧 Quick Start

### 1. Data Preparation
Ensure your dataset (`data/copolymer.csv`) contains polymer chain sequences and corresponding features:
- Chain sequences (variable length)
- Chemical composition (monomer ratios)
- Reaction conditions (temperature, time, etc.)
- Target block length distributions

### 2. Train Condition Encoder
```bash
python condition_encoder_trainer.py \
    --csv data/copolymer.csv \
    --out_dir outputs/condition_encoder \
    --epochs 100 \
    --batch_size 32 \
    --lr 1e-3
```

### 3. Full Training Pipeline with Monitoring
```bash
python prediction_eval.py \
    --csv data/copolymer.csv \
    --cond_ckpt outputs/condition_encoder/cond_encoder_final.pt \
    --out_dir outputs/full_training \
    --epochs 30 \
    --batch_size 8 \
    --accum_steps 2 \
    --sample_steps 200 \
    --guidance_scale 1.2 \
    --bins_new 50 \
    --fig_format pdf \
    --train_logit_temp 0.8 \
    --inference_logit_temp 1.4 \
    --fixed_test_indices "330,329,490,489,491,61,287,492,320,493,331,542"
```

### 4. Inference-Only Mode
```bash
python prediction_eval.py \
    --csv data/copolymer.csv \
    --cond_ckpt outputs/condition_encoder/cond_encoder_final.pt \
    --load_model outputs/full_training/best_model.pt \
    --inference_only \
    --out_dir outputs/inference_results \
    --sample_steps 200 \
    --guidance_scale 1.2 \
    --bins_new 50 \
    --fixed_test_indices "330,329,490,489,491,61,287,492,320,493,331,542"
```

## 📊 Output Structure

After training, the output directory will contain:

```
outputs/your_run/
├── ckpts/                       # Model checkpoints
│   ├── dit_epoch01.pt
│   ├── dit_epoch02.pt
│   └── ...
├── best_model.pt               # Best model (lowest test KL)
├── overlays/                   # Individual test sample overlays
│   ├── test_00330.pdf
│   ├── test_00329.pdf
│   └── ...
├── epoch_overlays/             # Training progress visualization
│   ├── epoch_01/
│   ├── epoch_02/
│   └── ...
├── top12_grid.pdf              # Final top-k comparison grid
├── top12_grid_epoch01.pdf      # Epoch-wise top-k grids
├── top12_grid_epoch02.pdf
└── test_rankings.csv           # Detailed performance rankings
```

## 🎯 Key Parameters

### Model Architecture
- `--dit_d_model`: Transformer hidden dimension (default: 256)
- `--dit_layers`: Number of transformer layers (default: 8)
- `--dit_heads`: Number of attention heads (default: 8)

### Training Control
- `--epochs`: Number of training epochs
- `--batch_size`: Training batch size
- `--accum_steps`: Gradient accumulation steps
- `--lr`: Learning rate (default: 2e-4)

### Diffusion Parameters
- `--T`: Total diffusion timesteps (default: 1000)
- `--sample_steps`: DDIM sampling steps (default: 200)
- `--guidance_scale`: Classifier-free guidance strength (default: 1.2)
- `--train_logit_temp`: Training temperature (default: 0.8)
- `--inference_logit_temp`: Inference temperature (default: 1.4)

### Data & Evaluation
- `--bins_new`: Histogram bin count (default: 50)
- `--fixed_test_indices`: Fixed test set indices for consistent evaluation
- `--top_k`: Number of best samples in visualization grid (default: 12)
- `--metric`: Ranking metric (kl/emd/combined)

## 🧪 Advanced Features

### Mayo-Lewis Theory Integration
All visualizations include theoretical Mayo-Lewis curves for validation:
- Red dashed lines with circle markers
- Automatic calculation from polymer sequence composition
- Handles complex copolymerization kinetics

### Real-Time Training Monitoring
- Epoch-wise test set evaluation
- Automatic best model selection based on KL divergence
- Progressive visualization generation
- Detailed logging with performance tracking

### Flexible Evaluation Modes
1. **Full Training**: Train from scratch with monitoring
2. **Continue Training**: Load pre-trained model and continue
3. **Inference Only**: Direct evaluation without training

## 📈 Performance Metrics

The system evaluates models using:
- **KL Divergence**: Primary metric for distribution comparison
- **Earth Mover's Distance (EMD)**: Alternative distribution metric
- **Combined Score**: Normalized combination of KL + EMD

## 🔍 Troubleshooting

### Common Issues

**Memory Issues**:
- Reduce `--batch_size` or increase `--accum_steps`
- Use `--cpu` flag for CPU-only training

**Training Stability**:
- Adjust `--train_logit_temp` (lower for more stability)
- Modify `--lr` and `--weight_decay`

**Generation Quality**:
- Tune `--guidance_scale` (1.0-2.0 range)
- Adjust `--inference_logit_temp`
- Increase `--sample_steps` for better quality


## 📚 References

This project builds upon:
- [DiT: Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748)
- [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502)
- Mayo-Lewis Copolymerization Theory


## 🙏 Acknowledgments

- Contributors to the polymer science and machine learning communities
- PyTorch team for the excellent deep learning framework
- The diffusion models research community

---

For questions or support, please open an issue or contact the maintainers.
