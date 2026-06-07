# Computer-Vision-Project-1
3D reconstruction from 2D brain scans

## Overview

This project investigates image registration and image processing techniques for reconstructing a three-dimensional representation of a brain from a series of two-dimensional histological slices.

The dataset consists of unregistered brain scans acquired from consecutive tissue sections. Since each slice may contain small positional shifts, rotations, or deformations, the images must first be aligned before they can be combined into a coherent 3D volume.

The project focuses on implementing the fundamental algorithms required for this reconstruction pipeline, including geometric transformations, image interpolation, histogram equalization, filtering, and anti-aliased downsampling.

---

## Objectives

The main goals of this project are:

* Visualize and analyze brain slice images
* Estimate geometric transformations between neighboring slices
* Register images using manually selected landmarks
* Implement image processing algorithms from scratch
* Improve image quality through filtering and contrast enhancement
* Prepare the dataset for future 3D reconstruction

---

## Dataset

The dataset contains:

* 27 grayscale TIFF images representing consecutive brain slices
* Landmark correspondences for image registration
* Example outputs for validation

```text
data/
├── tiff/
│   ├── B20_0429.tif
│   ├── B20_0430.tif
│   ├── ...
│
├── keypoints_ex1/
│   ├── points_src.npy
│   └── points_dst.npy
```

Each image corresponds to a single histological section of the brain.

---

## Technologies Used

* Python
* NumPy
* Matplotlib
* OpenCV
* Pillow
* tifffile
* scikit-image
* Napari

---

## Exercise 1 — Image Registration

### Image Exploration

All TIFF images are loaded and visualized to gain an understanding of the dataset.

For each image, the following statistics are computed:

* Data type
* Image dimensions
* Minimum intensity
* Maximum intensity
* Mean intensity
* Median intensity


### Affine Transformation Estimation

An affine transformation is estimated from the corresponding landmark pairs.

The affine model allows:

* Translation
* Rotation
* Scaling
* Shearing

The transformation matrix is computed using a least-squares formulation and subsequently applied to align the source image with the target image.

---

### Backward Mapping

To generate the transformed image, backward mapping is implemented.

Instead of mapping source pixels directly into the destination image, each destination pixel is mapped back to its corresponding source coordinate.

Benefits:

* Prevents holes in the output image
* Produces smoother transformations
* Supports interpolation naturally

Pipeline:

```text
Destination Pixel
        ↓
Inverse Transformation
        ↓
Source Coordinate
        ↓
Interpolation
        ↓
Pixel Intensity
```

---


## Exercise 2 — Image Processing

### Bilinear Downsampling

A custom bilinear interpolation algorithm was implemented to reduce image resolution while preserving image content.

Downsampling factors tested:

```python
[2, 4, 8, 16]
```

Advantages:

* Reduced computational cost
* Lower memory consumption
* Faster processing for large datasets

---

### Histogram Equalization

Histogram equalization was implemented from scratch to improve image contrast.

The procedure:

1. Compute image histogram
2. Compute cumulative distribution function (CDF)
3. Generate intensity mapping
4. Apply mapping to image pixels

---

### Histogram Equalization on Downsampled Images

Instead of computing the histogram directly from the full-resolution image, the histogram is estimated from a downsampled version and then applied to the original image.

Advantages:

* Lower computational complexity
* Reduced memory usage
* Less sensitivity to high-frequency noise
* More stable intensity mapping

One observed limitation is that the images contain large background regions. Global histogram equalization therefore tends to amplify background noise due to the dominance of low-intensity pixels.

---

### Image Filtering

Several filters were implemented and compared.

#### Mean Filter

Computes the average intensity within a local neighborhood.

Properties:

* Reduces random noise
* Produces noticeable blurring

---

#### Median Filter

Replaces each pixel by the median value within a neighborhood.

Properties:

* Preserves edges better than the mean filter
* Effective against impulse noise

---

#### Gaussian Filter

Applies weighted averaging using a Gaussian kernel.

Properties:

* Smooths noise
* Produces natural-looking blur
* Preserves structures better than a mean filter

---

#### Sobel Filter

Computes image gradients in horizontal and vertical directions.

Properties:

* Detects edges
* Highlights anatomical boundaries
* Useful for feature extraction

---

### Anti-Aliased Downsampling

The Gaussian filter was integrated into the downsampling pipeline.

Process:

```text
Original Image
       ↓
Gaussian Filtering
       ↓
Downsampling
       ↓
Anti-Aliased Image
```

This reduces aliasing artifacts and preserves image quality during resolution reduction.

---
## Exercise 3 — Multi-Slice Registration

After validating the registration procedure on a pair of neighboring slices, the estimated transformations are extended to the complete image stack.

The goal is to align all brain slices into a common coordinate system by sequentially registering adjacent images. This step reduces inter-slice misalignment and prepares the dataset for volumetric reconstruction.

---

## Exercise 4 — 3D Reconstruction

The final stage of the project consists of reconstructing a three-dimensional brain volume from the registered two-dimensional slices.

The aligned image stack is combined into a volumetric representation that can be visualized and analyzed in 3D.

--- 

## Results

The resulting workflow successfully transforms a collection of unregistered histological slices into an aligned image stack using napari.



---


## Final Remarks

This project is a part of a Computer Vision course. This repository contains coursework developed for academic purposes. The code is provided as-is without any warranty of reliability, or fitness for a particular purpose.
