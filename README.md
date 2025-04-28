# Deep Generative CNN Classifiers with Latent Diffusion

This project implements a complete pipeline for generating new CNN classifiers by learning a latent space of network parameters using an autoencoder and a latent diffusion model. The models are trained on the MNIST dataset.

## Project Overview

1. **Train multiple CNN classifiers** to collect parameter datasets.
2. **Train a Parameter Autoencoder** to encode high-dimensional CNN parameters into a compact latent space.
3. **Train a Latent Diffusion Model** on the latent space to model the parameter distribution.
4. **Sample new CNN parameters** using DDIM-based sampling from the diffusion model.
5. **Evaluate** the generated CNN classifiers on MNIST test set.
6. **Visualize and Save** training losses and accuracies for each stage.

---

## Project Structure

- `CNNClassifier`: Basic convolutional neural network for MNIST classification.
- `ParameterAutoencoder`: Autoencoder that compresses flattened CNN parameters.
- `LatentDiffusion`: Diffusion model that operates on the latent space learned by the autoencoder.
- `update_ema()`: Exponential Moving Average update for stabilizing diffusion model during training.
- `main()`: The complete training, sampling, evaluation, and visualization pipeline.

---

## Dependencies

Install required libraries:

```bash
pip install torch torchvision matplotlib pandas tqdm
```

---

## How to Run

Simply execute the main script:

```bash
python main.py
```

The code will:
- Train 10 CNNs on MNIST for 10 epochs each.
- Train an autoencoder on the collected CNN parameters.
- Train a diffusion model for 20,000 steps.
- Generate 20 new CNN models.
- Test and report accuracy of generated models.
- Plot and save:
  - CNN training loss curve
  - Autoencoder training loss curve
  - Diffusion training loss curve
  - Generated CNN accuracy scatter plot
- Save:
  - `training_curves_TIMESTAMP.png`
  - `training_metrics_TIMESTAMP.csv`

Both files are stored in `output/plots/`.

---

## Visualized Outputs

- **CNN Training Loss Curve**: Monitors CNN training process.
- **Autoencoder Loss Curve**: Tracks parameter reconstruction ability.
- **Diffusion Loss Curve**: Shows how well the diffusion model fits the latent distribution.
- **Generated CNN Test Accuracies**: Shows the performance of generated CNN models.

Example output files:
```
output/plots/
    training_curves_20250101_0000.png
    training_metrics_20250101_0000.csv
```

---

## Key Parameters (Default Settings)

| Parameter | Value |
|:---|:---|
| Number of CNNs | 10 |
| CNN Epochs | 10 |
| Autoencoder Latent Dim | 512 |
| Autoencoder Hidden Dim | 1024 |
| Autoencoder Epochs | 100 |
| Diffusion Timesteps | 500 |
| Diffusion Training Steps | 20,000 |
| Generated CNNs | 20 |
| Diffusion Sampling Steps (DDIM) | 50 |

---

## Notes

- **Losses and accuracies are automatically logged and visualized**.
- **Generated CNN models are fully functional** and tested on MNIST.
- **All outputs are timestamped**, ensuring experiments are well organized.

---

## Future Work (Optional Extensions)

- Apply to larger datasets (e.g., CIFAR-10, CIFAR-100).
- Introduce Classifier-Free Guidance for conditional generation.
- Improve sampling with DPM-Solver or EDM.
- Fine-tune generated CNNs for higher accuracy.
- Ensemble multiple generated CNNs for better stability.

---
