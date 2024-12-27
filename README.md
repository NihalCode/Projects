Mirrored Strategy Project

This code trains a convolutional neural network (CNN) to classify CIFAR-10 images, leveraging TensorFlow’s MirroredStrategy for distributed training across multiple GPUs. MirroredStrategy replicates the model across devices, 
synchronizing gradients to ensure efficient parallelism, which accelerates training. CIFAR-10 data is normalized and passed to a model with three convolutional layers, dropout for regularization, and dense layers for classification. 
The model is compiled with the Adam optimizer and sparse categorical cross-entropy loss. Training runs twice—once for 15 epochs using MirroredStrategy, and another 5 epochs without it, 
allowing comparison between distributed and single-device performance.
