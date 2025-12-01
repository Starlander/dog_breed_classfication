# Dog Breed Classification with EfficientNetV2S + CBAM

A deep learning pipeline for classifying 120 breeds of dogs from images, using TensorFlow, EfficientNetV2S, and the CBAM attention module.  
This approach leverages state-of-the-art architectures and training strategies for high accuracy and generalizability.

## Table of Contents

- [Overview](#overview)
- [Pipeline](#pipeline)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Model Architecture](#model-architecture)
- [Usage](#usage)
- [References](#references)

---

## Overview

This repository implements a robust image classification pipeline to identify dog breeds, built upon EfficientNetV2S and enhanced by CBAM (Convolutional Block Attention Module).  
The training is performed on the [Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/) with strong augmentation and regularization for real-world performance.

## Pipeline

1. **Preprocessing & Augmentation:**  
   - Random horizontal flips, contrast, brightness, and zoom.
   - Dataset split into train/validation/test.
   - Images resized to 224x224.

2. **Model:**  
   - EfficientNetV2S as backbone (pretrained on ImageNet), with CBAM attention block.
   - Mixed precision training for speed and efficiency.
   - Label smoothing to improve calibration.

3. **Training Procedure:**  
   - Two-phase ("warmup" and "finetune"):  
     - Warmup: Train classifier head with backbone frozen for several epochs.  
     - Finetune: Unfreeze top backbone blocks and use cosine-decay learning rate.
   - AdamW optimizer with weight decay and scheduled learning rate.
   - Early stopping and checkpointing based on validation accuracy.

4. **Evaluation:**  
   - Metrics include accuracy, top-5 accuracy, and macro F1-score.
   - Results on train/validation/test sets.

## Installation

Clone this repository and install dependencies:
```bash
git clone https://github.com/Starlander/dog_breed_classfication.git
cd dog_breed_classfication
pip install -r requirements.txt
```

## Dataset Preparation

- Download and extract the Stanford Dogs Dataset and split into `train/`, `val/`, and `test/` folders under `data/Images_split/`.
- You can use provided scripts/notebook cells for directory structure and split.

Expected directory tree:
```
data/
  Images_split/
    train/
    val/
    test/
```

## Training

Run the notebook `DogBreedClassification_EfficientNET_CBAM_FINAL_224.ipynb` on Jupyter/Kaggle with GPU enabled.
- Change `DATA_ROOT` in the notebook to your local dataset path as needed.
- The training pipeline includes:
  - Model instantiation with EfficientNetV2S and CBAM.
  - Mixed precision policy.
  - Warmup (train head only), followed by selective fine-tuning.
  - Checkpoints and TensorBoard logs.

Key training hyperparameters:
- Image size: 224x224
- Batch size: 16
- Classes: 120
- Warmup epochs: 5
- Fine-tune epochs: 30
- Learning rates: 1e-4 (warmup), 2e-5 (finetune, cosine-decayed)
- Label smoothing: 0.1

## Evaluation

The notebook automatically evaluates and prints metrics on validation and test sets.  
Metrics include accuracy, top-5 accuracy, and F1-score.

Example output:
```
Train: loss=0.9271, acc=0.9870, top5=0.9999, F1=0.9876
Validation: loss=1.0731, acc=0.9326, top5=0.9939, F1=0.9316
Test: loss=1.0848, acc=0.9254, top5=0.9947, F1=0.9239
```

## Results

| Set        | Accuracy | Top-5 Acc | Macro F1 |
|------------|----------|-----------|----------|
| Train      | 98.70%   | 99.99%    | 0.9876   |
| Validation | 93.26%   | 99.39%    | 0.9316   |
| Test       | 92.54%   | 99.47%    | 0.9239   |

- The final model is saved as: `artifacts/effv2s_cbam_final.keras`
- Checkpoints are saved during training in the `checkpoints/` folder.

## Model Architecture

- **Backbone:** EfficientNetV2S (ImageNet pretrained, partially fine-tuned)
- **Attention:** CBAM (Convolutional Block Attention Module) integrated after backbone.
- **Head:** Global Average Pooling → BatchNorm → Dropout → Dense (Softmax)
- **Regularization:** Weight decay, label smoothing, dropout, data augmentation.

Model summary:
```
Input (224x224x3)
  └── EfficientNetV2S backbone
       └── CBAM block
           └── GlobalAveragePooling2D
               └── BatchNormalization
                   └── Dropout
                       └── Dense(120, softmax)
```

## Usage

You can use the saved model for inference on new images as follows (example code snippet):

```python
import tensorflow as tf

model = tf.keras.models.load_model('artifacts/effv2s_cbam_final.keras')

img = tf.keras.utils.load_img('path/to/dog.jpg', target_size=(224, 224))
x = tf.keras.utils.img_to_array(img)
x = tf.expand_dims(x, axis=0)
x = efficientnet_v2.preprocess_input(x)

probs = model.predict(x)
breed_idx = probs.argmax(axis=1)[0]
confidence = probs[0, breed_idx]
print(f'Detected breed index: {breed_idx} with confidence {confidence:.3f}')
```

## References

- [EfficientNetV2: Smaller Models and Faster Training](https://arxiv.org/abs/2104.00298)
- [CBAM: Convolutional Block Attention Module](https://arxiv.org/abs/1807.06521)
- [Stanford Dog Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/)

---

*Author: [Starlander](https://github.com/Starlander)*  
*Contact: tuananh141012@gmail.com*
