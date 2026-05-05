# Aerial Image Segmentation with PyTorch U-Net

## Project Overview

This project implements a binary semantic segmentation workflow for extracting road networks from aerial imagery using PyTorch. The notebook trains a U-Net based segmentation model to classify each pixel in an aerial image as either road or non-road, producing a single-channel binary mask aligned with the input image.

The implementation is centered on an image-mask learning pipeline: aerial RGB images are paired with grayscale ground-truth road masks, both are transformed consistently with segmentation-aware augmentations, and a convolutional neural network is optimized to predict road masks from image content. The model uses the `segmentation-models-pytorch` library to build a U-Net architecture with an ImageNet-pretrained `timm-efficientnet-b0` encoder.

The full workflow is contained in the notebook `aerial_segmentation_unet_pytorch.ipynb`. It covers dataset inspection, preprocessing, augmentation, custom dataset creation, batching, model definition, training, validation, checkpointing, and inference visualization.

## Objectives

The project is designed around the following technical objectives:

- Understand the structure of an image-mask segmentation dataset.
- Use Albumentations to apply transformations jointly to images and masks.
- Build a custom PyTorch `Dataset` for paired aerial images and road masks.
- Load a pretrained convolutional neural network backbone for a segmentation task.
- Construct a U-Net based binary segmentation model.
- Write reusable training and validation loops in PyTorch.
- Run inference by converting raw model logits into thresholded binary masks.
- Visualize input images, ground-truth masks, and predicted masks for qualitative evaluation.

## Dataset

The notebook uses a subset of the Massachusetts Roads Dataset. The original dataset contains aerial imagery of Massachusetts and corresponding road labels. Each original image is `1500x1500` pixels and covers approximately `2.25` square kilometers.

The subset is loaded from:

```text
https://github.com/parth1620/Road_seg_dataset.git
```

The notebook describes this subset as approximately 200 image-mask pairs. The executed notebook output shows that the loaded `train.csv` file contains `199` rows with two columns:

| Column | Description |
| --- | --- |
| `images` | Relative path to an aerial RGB image. |
| `masks` | Relative path to the corresponding road mask. |

Example row pattern:

```text
images/17428750_15.png    masks/17428750_15.png
```

The dataset is split with `train_test_split` using `test_size=0.2` and `random_state=42`, producing:

| Split | Samples |
| --- | ---: |
| Training | `159` |
| Validation | `40` |
| Total | `199` |

The task is binary segmentation. Road pixels are represented in the mask, while background pixels represent non-road regions. During dataset loading, masks are read as grayscale images, expanded to one channel, scaled to `[0, 1]`, and rounded to ensure binary labels.

## Technical Stack

| Component | Role in the project |
| --- | --- |
| Python | Main programming language used in the notebook. |
| PyTorch | Tensor computation, dataset abstraction, model definition, optimization, and training loop execution. |
| segmentation-models-pytorch | Provides the U-Net implementation and pretrained encoder integration. |
| timm | Supplies the EfficientNet encoder used by `segmentation-models-pytorch`. |
| Albumentations | Applies image-mask augmentations for segmentation data. |
| OpenCV | Reads images and masks and converts image color channels from BGR to RGB. |
| NumPy | Handles array reshaping, channel expansion, transposition, and numeric preprocessing. |
| Pandas | Loads and inspects the CSV file containing image-mask paths. |
| scikit-learn | Creates the train-validation split. |
| Matplotlib | Displays input images, ground-truth masks, and predictions. |
| tqdm | Shows progress bars during training and validation loops. |

The notebook begins by installing or upgrading these primary packages:

```text
segmentation-models-pytorch
albumentations
opencv-contrib-python
```

The implementation is designed around a GPU runtime, with `DEVICE = 'cuda'` in the configuration cell.

## Model Architecture

The segmentation model is implemented as a custom PyTorch `nn.Module` named `SegmentationModel`. Internally, it wraps a U-Net from `segmentation-models-pytorch`:

```python
smp.Unet(
    encoder_name='timm-efficientnet-b0',
    encoder_weights='imagenet',
    in_channels=3,
    classes=1,
    activation=None
)
```

Key architectural choices:

| Setting | Value |
| --- | --- |
| Architecture | U-Net |
| Encoder | `timm-efficientnet-b0` |
| Encoder initialization | ImageNet pretrained weights |
| Input channels | `3` RGB channels |
| Output channels | `1` binary mask channel |
| Output activation inside model | `None` |

The model returns raw logits rather than probabilities. This is important because the training objective includes `BCEWithLogitsLoss`, which expects unnormalized logits and internally applies the sigmoid operation in a numerically stable way.

When masks are passed into the model during training or validation, the forward method returns both:

- `logits`: raw per-pixel model outputs.
- `loss`: combined segmentation loss.

When masks are not passed, the model returns only `logits`, which is the behavior used during inference.

## Project Workflow

### 1. GPU Runtime and Package Setup

The notebook starts by preparing a Colab-style GPU environment. It installs `segmentation-models-pytorch`, installs `albumentations`, and upgrades `opencv-contrib-python`. The project configuration then sets `DEVICE = 'cuda'`, indicating that tensors and the model are expected to run on a CUDA-enabled GPU.

### 2. Dataset Download and Path Configuration

The road segmentation subset is cloned from the `Road_seg_dataset` GitHub repository. The dataset path is added to `sys.path` so the notebook can import the provided `helper` module for visualization.

The main dataset configuration is:

```python
CSV_FILE = '/content/Road_seg_dataset/train.csv'
DATA_DIR = '/content/Road_seg_dataset/'
```

The CSV file stores relative image and mask paths. Each full path is created by concatenating `DATA_DIR` with the corresponding CSV entry.

### 3. Dataset Inspection and Visualization

The CSV is loaded with Pandas, and sample image-mask pairs are selected by row index. Images are loaded with OpenCV, converted from BGR to RGB, and displayed beside their ground-truth masks.

For initial inspection, the notebook visualizes examples such as:

- `idx = 2`
- `idx = 15`

This confirms that the input imagery and road labels are spatially aligned before model training begins.

### 4. Train-Validation Split

The dataset is split into training and validation DataFrames:

```python
train_df, valid_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42
)
```

The resulting split contains `159` training samples and `40` validation samples. The fixed random seed makes the split reproducible across notebook runs.

### 5. Segmentation Augmentation

Albumentations is used because segmentation tasks require identical geometric transformations to be applied to both the image and its mask. Applying image-only augmentations would break the pixel-level correspondence between the aerial image and ground-truth mask.

Training augmentations:

```python
A.Resize(512, 512)
A.HorizontalFlip(p=0.5)
A.VerticalFlip(p=0.5)
```

Validation augmentations:

```python
A.Resize(512, 512)
```

The training pipeline uses horizontal and vertical flips to increase spatial diversity while preserving road-mask alignment. The validation pipeline only resizes data so validation performance is measured on deterministic transformed samples.

### 6. Custom PyTorch Dataset

The notebook defines `SegmentationDataset`, a subclass of `torch.utils.data.Dataset`, to load paired images and masks.

For each sample, the dataset performs the following operations:

1. Select one row from the DataFrame.
2. Build the image path and mask path.
3. Read the image with OpenCV.
4. Convert the image from BGR to RGB.
5. Read the mask as grayscale.
6. Expand the mask shape to include a channel dimension.
7. Apply Albumentations transforms to both image and mask.
8. Transpose image and mask from channel-last format to channel-first format.
9. Convert arrays to `float32`.
10. Convert arrays to PyTorch tensors.
11. Scale image pixels by dividing by `255.0`.
12. Scale and round mask pixels to produce binary labels.

The final per-sample tensor shapes are:

| Tensor | Shape | Meaning |
| --- | --- | --- |
| Image | `[3, 512, 512]` | RGB image tensor |
| Mask | `[1, 512, 512]` | Binary road mask tensor |

The notebook creates separate dataset instances:

```python
trainset = SegmentationDataset(train_df, get_train_augs())
validset = SegmentationDataset(valid_df, get_valid_augs())
```

Sample augmented image-mask pairs are then visualized with `helper.show_image`.

### 7. Batch Loading

The datasets are wrapped in PyTorch `DataLoader` objects:

```python
trainloader = DataLoader(trainset, batch_size=8, shuffle=True)
validloader = DataLoader(validset, batch_size=8)
```

The training loader shuffles samples to reduce ordering bias during optimization. The validation loader does not shuffle because validation is used only for evaluation.

The executed notebook confirms:

| Loader | Batches |
| --- | ---: |
| Training loader | `20` |
| Validation loader | `5` |

One training batch has the following shapes:

| Batch tensor | Shape |
| --- | --- |
| Images | `[8, 3, 512, 512]` |
| Masks | `[8, 1, 512, 512]` |

### 8. Loss Function Design

The model combines two losses:

```python
DiceLoss(mode='binary')(logits, masks) + nn.BCEWithLogitsLoss()(logits, masks)
```

This pairing is useful for binary segmentation:

- `DiceLoss` directly optimizes spatial overlap between predicted road regions and ground-truth masks.
- `BCEWithLogitsLoss` provides stable per-pixel binary classification supervision from raw logits.

The combined objective balances region-level mask quality with pixel-level classification accuracy.

### 9. Training Function

The `train_fn` routine handles one full training epoch. It switches the model to training mode, moves images and masks to the configured device, clears gradients, computes logits and loss, backpropagates the loss, updates model weights with the optimizer, and accumulates the average epoch loss.

The optimizer used in the notebook is:

```python
torch.optim.Adam(model.parameters(), lr=0.003)
```

### 10. Validation Function

The `eval_fn` routine evaluates the model on the validation loader. It switches the model to evaluation mode and wraps inference in `torch.no_grad()` to disable gradient tracking. This reduces memory usage and ensures validation does not update model parameters.

The function returns the average validation loss across all validation batches.

### 11. Checkpointing

The training loop runs for `15` epochs. After each epoch, the validation loss is compared against the best validation loss seen so far. If the current validation loss is lower, the model weights are saved:

```python
torch.save(model.state_dict(), "best-model.pt")
```

This checkpointing strategy keeps the best-performing model according to validation loss rather than simply using the final epoch weights.

### 12. Inference and Mask Prediction

For inference, the notebook loads the saved checkpoint and selects validation samples by index. A single image tensor is moved to the GPU and expanded with `unsqueeze(0)` to create a batch dimension.

The model outputs raw logits:

```python
logits_mask = model(image.to(DEVICE).unsqueeze(0))
```

The logits are converted to probabilities with sigmoid:

```python
pred_mask = torch.sigmoid(logits_mask)
```

The probability mask is thresholded at `0.5`:

```python
pred_mask = (pred_mask > 0.5) * 1.0
```

The final predicted binary mask is visualized beside the original image and ground-truth mask using `helper.show_image`.

## Training Configuration

| Parameter | Value |
| --- | --- |
| Device | `cuda` |
| Epochs | `15` |
| Learning rate | `0.003` |
| Optimizer | Adam |
| Batch size | `8` |
| Image size | `512x512` |
| Encoder | `timm-efficientnet-b0` |
| Encoder weights | ImageNet |
| Model architecture | U-Net |
| Loss | Binary Dice loss + BCE with logits |
| Train samples | `159` |
| Validation samples | `40` |
| Train batches | `20` |
| Validation batches | `5` |

## Training Results

The notebook logs training and validation loss for all 15 epochs:

| Epoch | Training Loss | Validation Loss | Checkpoint Saved |
| ---: | ---: | ---: | --- |
| 1 | `1.2744326382875442` | `1.0927199363708495` | Yes |
| 2 | `0.8240275502204895` | `0.9125379920005798` | Yes |
| 3 | `0.7304657340049744` | `0.7270325899124146` | Yes |
| 4 | `0.7021044552326202` | `0.7907907366752625` | No |
| 5 | `0.6896344214677811` | `0.7029729604721069` | Yes |
| 6 | `0.6564587175846099` | `0.7359623670578003` | No |
| 7 | `0.648715540766716` | `0.6900284886360168` | Yes |
| 8 | `0.6366149514913559` | `0.6902047276496888` | No |
| 9 | `0.6367130607366562` | `0.6609917402267456` | Yes |
| 10 | `0.6286943942308426` | `0.6469156742095947` | Yes |
| 11 | `0.6378325462341309` | `0.6884613633155823` | No |
| 12 | `0.613987585902214` | `0.645119559764862` | Yes |
| 13 | `0.6067466050386429` | `0.6529337525367737` | No |
| 14 | `0.6136820331215859` | `0.6772048354148865` | No |
| 15 | `0.6149189651012421` | `0.6537615180015564` | No |

The best validation loss observed in the notebook is:

```text
0.645119559764862 at epoch 12
```

Training loss decreases sharply during the first few epochs, indicating that the pretrained encoder and U-Net decoder quickly adapt to the road segmentation task. Validation loss improves overall but fluctuates after the early epochs, which is expected for a small segmentation subset with limited training samples.

## Inference Pipeline

The inference stage uses the best checkpoint saved during training. The notebook repeatedly evaluates validation samples at selected indices such as `15`, `14`, `30`, `20`, `22`, and `28`.

The prediction sequence is:

1. Load the trained model weights from `best-model.pt`.
2. Retrieve an image and mask from the validation dataset.
3. Add a batch dimension to the image tensor.
4. Run a forward pass through the U-Net model.
5. Apply sigmoid to convert logits into probabilities.
6. Apply a `0.5` threshold to produce a binary mask.
7. Move the prediction back to CPU for visualization.
8. Display the input image, ground-truth mask, and predicted mask together.

This inference flow keeps the model output interpretable by converting continuous logits into a clear road/non-road segmentation map.

## Key Takeaways

- The project demonstrates an end-to-end semantic segmentation pipeline for aerial road extraction.
- Image-mask alignment is preserved by using Albumentations transforms that operate on both inputs together.
- A custom PyTorch dataset handles image loading, RGB conversion, mask loading, channel formatting, normalization, and binarization.
- The U-Net model uses an ImageNet-pretrained EfficientNet encoder, reducing the amount of task-specific data required to learn useful visual features.
- The loss combines Dice overlap optimization with stable pixel-wise binary classification.
- The checkpointing logic selects the model with the best validation loss, which occurred at epoch `12`.
- Inference uses sigmoid activation and a `0.5` threshold to convert raw model logits into final binary road masks.

## Dataset Citation

The original Massachusetts Roads Dataset is associated with the following thesis:

```bibtex
@phdthesis{MnihThesis,
  author = {Volodymyr Mnih},
  title = {Machine Learning for Aerial Image Labeling},
  school = {University of Toronto},
  year = {2013}
}
```

Full dataset source:

```text
https://www.cs.toronto.edu/~vmnih/data/
```

Subset used in the notebook:

```text
https://github.com/parth1620/Road_seg_dataset
```
