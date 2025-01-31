# Denoising_HPG_MRI

# Code Overview

1. Data Loading and Preprocessing:

-The MRI data is loaded using the nibabel library.
-The data is normalized to ensure that pixel values are in the range [0, 1].
-Noise is added to the clean images to create noisy input data for training.

2. Model Architecture:

-The model is based on a U-Net architecture, which is commonly used for image segmentation and denoising tasks.
-Two versions of the U-Net are provided: one with skip connections and one without.
-The model is compiled using the Adam optimizer and binary cross-entropy loss.

3. Training:

-The model is trained on noisy and clean image pairs.
-The training process is iterative, with the model weights being updated after each epoch.
-The learning rate is reduced over time to improve convergence.

4. Evaluation and Visualization:

-After training, the model is used to predict denoised images from the test set.
-The results are visualized using Matplotlib, showing the original, noisy, and denoised images side by side.

## Training  Explanation

# Overview
- Processes 500 slices at a time, adding noise before training.
- Trains a deep learning model iteratively with progressive data chunks.
- Shuffles data after all slices are processed.
- Reduces learning rate over time to fine-tune the model.

1. Initialization
a = 0, b = 499: Defines the range of slices processed in each iteration (500 slices at a time).
LR = 0.01: Sets the initial learning rate for training.

2. Iterating Over Data in Parts
The loop runs 9 times (for part in range(1, 10):).
Each iteration processes a new subset of the dataset.

3.Processing a Data Subset
Prints the current range of slices: print('a', a, 'b', b).
Calls noiseAddinForLoop(clean[:, :, a:b]):
 Adds noise to the selected slices.
 Outputs:
    - oo: Clean images (ground truth).
    - nn: Noisy images (input for the model).

4.Splitting Data for Training
Uses train_test_split(nn, oo, test_size=0.1):
90% training data (xtrain, ytrain).
10% test data (xtest, ytest).

5. Restoring Model Weights & Compiling
model.set_weights(myWeights): Restores the previously trained model weights.
model.compile(keras.optimizers.Adam(learning_rate=LR), loss='binary_crossentropy', metrics=['accuracy']):
Adam optimizer with the current learning rate (LR).
Binary cross-entropy loss (suggesting a binary classification problem).
Tracks accuracy as a performance metric.

6. Training the Model
model.fit(xtrain, ytrain, epochs=10, verbose=1):
Trains the model on the new batch of noisy-clean image pairs.
Runs for 10 epochs.
Displays training progress (verbose=1).

7. Updating Weights & Moving to the Next Data Chunk
myWeights = model.get_weights(): Saves the updated weights after training.
a = b + 1, b = b + 499: Moves to the next set of slices for training.

8. Shuffling Data When All Slices Are Processed
If all slices have been processed (b > clean.shape[2]):
Shuffles slice indices
Creates a new empty array (clean_shuffle) for shuffled data.
Reorders slices based on the shuffled indices.
Updates clean = clean_shuffle to use the shuffled dataset.

9. Reducing the Learning Rate
LR = LR / 10:
Divides the learning rate by 10 to refine training.
Helps the model learn more gradually over time.

