# cse498repo
#### Team Name
Calcite  
#### Team Members
Junjiang Xiao; Hengde Dai  
#### Project Name
Study of Neural Network Parameter Diffusion based on image classification
#### Project Abstract
Diffusion models have recently achieved significant breakthroughs in generative artificial intelligence, excelling at generating high-quality images, audio, and text. However, their capabilities extend beyond content generation. This project explores a novel research direction—leveraging diffusion models to directly generate neural network parameters, reducing reliance on traditional gradient-based optimization and enhancing training efficiency.

Our project is to study the P-Diff (Neural Network Parameter Diffusion), a framework that treats high-performance neural network parameters as structured data. The key idea is to train a diffusion model to learn the distribution of these parameters in a latent noise space, enabling the direct synthesis of optimized network parameters. First, we collect parameters from multiple pre-trained high-performance neural networks (e.g., ResNet, ViT) and compress them using an autoencoder to obtain low-dimensional latent representations. Next, we train a diffusion model in this latent space to iteratively refine noise into structured parameters. Finally, a decoder reconstructs the generated latent vectors into deployable neural network parameters, enabling an end-to-end process from noise to optimized networks.

To validate P-Diff, we will conduct extensive experiments on public datasets such as CIFAR-10 and ImageNet. We will train CNN architectures like ResNet and EfficientNet and evaluate the performance of networks synthesized by P-Diff. Our benchmark will measure accuracy (Top-1/Top-5 accuracy, F1 score), generalization (cross-dataset testing), and computational efficiency (FLOPs, inference latency, memory footprint), comparing P-Diff-generated models against traditionally trained ones. Our goal is to explore the potential of diffusion models in neural network parameter generation and establish a new paradigm for neural network optimization.