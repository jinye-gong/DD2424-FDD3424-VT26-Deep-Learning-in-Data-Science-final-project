# DD2424 Project Proposal

**Project Proposal: Efficient ConvNet Design on CIFAR-10**  
**Group 148**  
**Boyi Shi, Puhao Zhu, Jinye Gong**  
**April 2026**

---

## 1. Project Overview

**Title:** Scaling Laws and Accuracy–Efficiency Trade-offs in Modern ConvNets on CIFAR-10

**Project Type:** Default Project 2 — Building and training a modern ConvNet architecture from scratch

**Target Grade:** A/B

Our project is based on Default Project 2. The basic part of the project will build and train a modern ConvNet architecture from scratch on CIFAR-10, starting from a VGG-style baseline and progressively improving the architecture and training setup.

For the A/B-level extension, we will focus on the directions **“ConvNet vs modern architectures”** and **“Modern architectural components”** through a scaling-law style study. In particular, we will compare a strong VGG-style ConvNet baseline with ConvNeXt-style architectural components under controlled computational budgets, and derive budget-aware design rules from systematic scaling experiments.

---

## 2. Problem and Research Question

Convolutional neural networks can achieve strong performance on CIFAR-10, but different architectural choices may lead to very different accuracy–efficiency trade-offs. Standard VGG-style networks are simple and effective, but they may use parameters and FLOPs inefficiently. More modern architectures, such as ConvNeXt-style models, use components such as large kernels, depthwise convolutions, LayerNorm, and inverted bottleneck-like structures.

However, CIFAR-10 images are only 32×32 pixels. It is therefore not obvious whether large-kernel ConvNet designs, originally motivated by larger-scale image recognition tasks, remain beneficial for such low-resolution images.

Our main research questions are:

1. **Do modern large-kernel ConvNet designs, such as ConvNeXt-style blocks, provide a better accuracy–FLOPs trade-off than strong VGG-style ConvNet baselines on CIFAR-10?**

2. **How do depth, width, kernel size, and block type scale with computational budget on CIFAR-10, and which scaling direction gives the best marginal gain per extra FLOP?**

---

## 3. Proposed Approach

### 3.1 Basic Project Component

We will first implement the required Default Project 2 baseline using PyTorch. This includes:

- rewriting the Assignment 3 network using PyTorch modules and automatic differentiation;
- building a VGG-style ConvNet baseline for CIFAR-10;
- adding regularization such as dropout, weight decay, data augmentation, and batch normalization;
- testing optimizer and learning-rate schedule choices such as AdamW and cosine annealing;
- evaluating the model using test accuracy, parameter count, and FLOPs.

This part establishes a strong baseline before moving to the A/B-level extension.

---

### 3.2 A/B-Level Extension

For the A/B-level extension, we will focus on a systematic architecture-efficiency and scaling-law analysis. The extension is designed to match the Default Project 2 A/B directions:

- **ConvNet vs modern architectures**
- **Modern architectural components**

Specifically, we will:

1. **Conduct scaling-law style studies across architecture dimensions**  
   We will systematically scale depth, width, and kernel size (e.g., 3×3, 5×5, 7×7, 11×11) and measure how performance changes with compute and parameter budgets.

2. **Compare different convolution/block types**  
   We will compare:
   - standard convolutional VGG-style blocks;
   - depthwise separable convolution blocks;
   - ConvNeXt-style blocks with large kernels and LayerNorm.

3. **Control computational budgets for fair comparison**  
   We will adjust model width, depth, and channel dimensions so that different architectures are compared under matched or comparable FLOPs and/or parameter budgets.

4. **Construct accuracy–efficiency Pareto frontiers**  
   We will plot accuracy against FLOPs (and parameters), and identify Pareto-optimal models in low-, medium-, and high-budget regions.

5. **Analyze large-kernel effectiveness and scaling behavior on low-resolution data**  
   We will discuss whether large-kernel ConvNet designs remain useful on 32×32 CIFAR-10 images, and characterize their marginal return per extra compute.

6. **Derive budget-aware ConvNet design rules**  
   From scaling trends, we will summarize practical architecture recommendations for different FLOPs/parameter regimes.

MixUp and label smoothing will not be treated as A/B-level contributions. They may be used only as optional training components if needed, but the main extension is the scaling-law and architecture-efficiency study.

---

## 4. Dataset and Implementation

We will use the standard CIFAR-10 dataset, consisting of 50,000 training images and 10,000 test images across 10 classes.

A validation split will be created from the training set for model selection and hyperparameter tuning.

The implementation will be done in PyTorch. We will implement our own model classes for:

- VGG-style baseline blocks;
- depthwise separable convolution blocks;
- ConvNeXt-style blocks;
- model variants with different kernel sizes and channel widths.

We will compute:

- classification accuracy;
- validation and test loss;
- number of trainable parameters;
- FLOPs or MACs;
- training throughput / time per epoch;
- accuracy–FLOPs trade-off curves.

---

## 5. Evaluation Plan

The evaluation will focus on both classification performance and computational efficiency.

### 5.1 Baselines

We will compare against:

- the Assignment 3-style baseline;
- a VGG-style ConvNet baseline;
- improved VGG-style variants with regularization and batch normalization.

### 5.2 Architecture Comparisons

We will compare:

- standard convolution vs depthwise separable convolution;
- small kernels vs large kernels;
- VGG-style blocks vs ConvNeXt-style blocks;
- models under comparable FLOPs or parameter budgets;
- scaling behavior across depth/width/kernel-size dimensions.

### 5.3 Metrics

The main metrics will be:

- test accuracy;
- validation accuracy;
- training and validation loss;
- parameter count;
- FLOPs;
- accuracy per FLOP;
- marginal accuracy gain under increasing compute;
- Pareto efficiency.

### 5.4 Analysis

The final report will analyze:

- which architectural components improve accuracy;
- which components improve efficiency;
- whether larger kernels are beneficial on CIFAR-10;
- whether ConvNeXt-style blocks outperform VGG-style blocks under fair computational constraints;
- which models form the accuracy–FLOPs Pareto frontier;
- which scaling direction (depth/width/kernel size/block type) is most compute-efficient at each budget level.

---

## 6. Milestones

### E-Level Milestone

- Implement the PyTorch CIFAR-10 training pipeline.
- Reproduce a basic VGG-style ConvNet baseline.
- Train and evaluate the baseline model.
- Report accuracy, loss, parameter count, and FLOPs.

### C/D-Level Supporting Work

- Add regularization and training improvements where useful.
- Test dropout, weight decay, data augmentation, batch normalization, and learning-rate scheduling.
- Use these results to establish a stronger baseline.

### A/B-Level Extension

- Implement depthwise convolution and ConvNeXt-style blocks.
- Run scaling studies across depth, width, and kernel size (3×3 to larger kernels such as 7×7 and 11×11).
- Compare standard convolution, depthwise convolution, and ConvNeXt-style blocks under matched FLOPs or parameter budgets.
- Construct accuracy–FLOPs/parameter Pareto frontiers for different budget regions.
- Quantify marginal gain per added compute and derive budget-aware design rules.
- Analyze whether modern large-kernel ConvNet designs are effective on CIFAR-10.

---

## 7. Group Workload

### Boyi Shi

- Design and implement the architecture blocks.
- Implement VGG-style, depthwise convolution, and ConvNeXt-style model variants.
- Help control model width and depth for fair FLOPs comparisons.

### Puhao Zhu

- Set up the evaluation framework.
- Compute FLOPs and parameter counts.
- Generate accuracy–FLOPs plots and Pareto frontier visualizations.
- Train selected baseline and comparison models.

### Jinye Gong

- Implement and organize the training pipeline.
- Handle optimizer, scheduler, validation, and logging.
- Run ablation experiments and summarize results.
- Contribute to the final report analysis and self-assessment.

---

## 8. Expected Outcome

We expect to produce a clear empirical study of scaling laws and accuracy–efficiency trade-offs for modern ConvNet designs on CIFAR-10.

The final project should answer whether ConvNeXt-style design choices, especially large kernels and depthwise convolutions, are useful for low-resolution image classification when compared fairly against strong VGG-style ConvNet baselines, and which architecture scaling direction is most effective under each compute budget.

The main expected deliverables are:

- a working PyTorch implementation;
- trained baseline and modern ConvNet variants;
- systematic ablation results;
- scaling-law style analysis across architecture dimensions;
- accuracy–FLOPs Pareto frontier plots;
- budget-aware ConvNet design recommendations;
- a final report focused mainly on the A/B-level scaling and architecture-efficiency extension.

---

## References

[1] K. He, X. Zhang, S. Ren, and J. Sun.  
Deep Residual Learning for Image Recognition.  
Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2016.

[2] Z. Liu, H. Mao, C.-Y. Wu, C. Feichtenhofer, T. Darrell, and S. Xie.  
A ConvNet for the 2020s.  
Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022.

[3] DD2424 Teachers and TAs.  
Default Project 2 — Building and training a modern ConvNet architecture from scratch.  
DD2424, 2026.
