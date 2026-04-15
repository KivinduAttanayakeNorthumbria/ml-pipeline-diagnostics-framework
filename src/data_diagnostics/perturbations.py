# Stage 1 experiment - faults injections
import numpy as np
import pandas as pd
from PIL import ImageFilter, ImageEnhance, Image

# Inject fault values into tabular data
def inject_missing_value_tab(X, fraction, seed):
    print(f"Injecting {fraction*100:.0f}% fault values")

    gen = np.random.RandomState(seed)
    X_injected = X.copy()

    # Number of values to remove
    total_values = X_injected.shape[0] * X_injected.shape[1]
    num_missing = int(total_values * fraction)

    # Select random column and row
    rn_col = gen.randint(0, X_injected.shape[1], size = num_missing)
    rn_row = gen.randint(0, X_injected.shape[0], size = num_missing)

    # Change into Nan
    for r,c in zip(rn_row,rn_col):
        X_injected.iloc[r, c] = np.nan

    # Fill NaN with column median value
    X_injected = X_injected.fillna(X_injected.median())

    actual_missing = num_missing / total_values
    print(f"Removed {num_missing} values from the data")

    return X_injected

# Inject class imbalance
def inject_class_imbalance_tab(X, y, ratio, seed):
    print(f"Inject class imbalance, keeping {ratio * 100:.0f}% of minority class")

    gen = np.random.RandomState(seed)

    # Find minority class
    unique, counts =  np.unique(y, return_counts = True)
    minority_class = unique[np.argmin(counts)]
    minority_count = counts.min()

    # Get minority class samples to keep
    keep_count = int(minority_count * ratio)

    # Get indices of minority and majority sample
    minority_indices = np.where(y == minority_class)[0]
    majority_indices = np.where(y != minority_class)[0]

    # Randomly pick minority samples to keep
    keep_indices = gen.choice(minority_indices, size = keep_count, replace = False)

    # Combine majority indices with the reduced minority indices
    all_indices = np.concatenate([majority_indices, keep_indices])
    all_indices = np.sort(all_indices)

    # Create the imbalanced dataset
    X_imbalanced = X.iloc[all_indices].reset_index(drop = True)
    y_imbalanced = y[all_indices]

    print(f"Original minority count: {minority_count}")
    print(f"New minority count: {keep_count}")
    print(f"Majority count: {len(majority_indices)}")

    return X_imbalanced, y_imbalanced

# Inject noise into tabular data
def inject_noise_tab(X, noise_level, seed):
    print(f"Injecting noise at level {noise_level}")

    gen = np.random.RandomState(seed)
    X_noised = X.copy()

    # Add noise proportional to column
    for col in X_noised.columns:
        col_range = X_noised[col].max() - X_noised[col].min()
        noise = gen.normal(0, noise_level * col_range, size = len(X_noised[col]))
        X_noised[col] = X_noised[col] + noise

    print(f"Noise added to {len(X_noised.columns)} ")

    return X_noised

# Apply blur to an image
def apply_blur_img(image, blur):
    return image.filter(ImageFilter.GaussianBlur(radius = blur))

# Apply brightness change to an image
def apply_brightness_img(image, brightness):
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(1 + brightness)


# Apply random pixel noise to an image
def apply_noise_img(image, noise_level, seed):
    gen = np.random.RandomState(seed)
    img_array = np.array(image).astype(np.float32)
    noise = gen.normal(0, noise_level * 255, size = img_array.shape)
    img_noise = np.clip(img_array + noise, 0, 255).astype(np.uint8)

    return Image.fromarray(img_noise)


