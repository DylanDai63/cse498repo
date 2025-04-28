Project Title
Generating Neural Network Parameters via Latent Diffusion Models

Project Description
This project aims to explore the feasibility of generating deep learning model parameters directly using latent diffusion models.
We first train multiple CNN classifiers on the MNIST dataset, compress their parameters using an autoencoder, and then train a diffusion model in the latent space to synthesize new CNN parameters without traditional training.

File Structure
LnameFname/
├── data/
│   └── readme_data.txt    # Instructions on dataset
├── ReadMe.txt             # This README file
├── main.py                # Main training and evaluation script
└── aux_1.py               # Supporting modules and model definitions

How to Obtain Data
- The project automatically downloads the MNIST dataset from torchvision.datasets.MNIST.
- If automatic download fails, manually download from:
  http://yann.lecun.com/exdb/mnist/
- Place the extracted files under the data/ directory.

(See data/readme_data.txt for detailed instructions.)

How to Run the Code
1. Environment Setup
Make sure you have the following packages installed:
- torch
- torchvision
- pandas
- tqdm
- matplotlib

Install them via pip:
pip install torch torchvision pandas tqdm matplotlib

2. Running the Project
From the root project directory (LnameFname/), run:

python main.py

The script will:
- Train multiple CNN classifiers on MNIST
- Train a parameter autoencoder
- Train a latent diffusion model
- Generate new CNN models
- Evaluate the performance of generated models
- Save training curves and metrics under the output/plots/ directory

Notes
- It is highly recommended to use a GPU environment (e.g., CUDA) for faster training.
- Outputs include training loss curves, accuracy plots, and generated CNN evaluation results.
