# Stage 1 experiment - faults injections
import numpy as np
from PIL import ImageFilter, ImageEnhance

# Inject fault values into tabular data
def inject_fault_value_tab(X, fraction, seed):
    print(f"Injecting {fraction * 100:.0f}% fault values")

    gen = np.random.RandomState(seed)
    # Get a copy of original before change
    X_injected = X.copy()

    # Number of values to remove
    total_values = X_injected.shape[0] * X_injected.shape[1]
    # Calculate number of data to remove
    num_values_to_remove = int(total_values * fraction)

    # Select list of random column and row within the boundary
    random_columns = gen.randint(0, X_injected.shape[1], size = num_values_to_remove)
    random_rows = gen.randint(0, X_injected.shape[0], size = num_values_to_remove)

    # Change into Nan by simulating real word missing data
    for r,c in zip(random_rows,random_columns):
        X_injected.iloc[r, c] = np.nan

    # Fill NaN with column median value ( median = robust to outliers)
    X_injected = X_injected.fillna(X_injected.median())

    print(f"Perturbed {fraction * 100:.0f} values from the data")
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
    for column in X_noised.columns:
        # Calculate range of the column ( max - min )
        column_range = X_noised[column].max() - X_noised[column].min()
        # Calculate noise value and generate noise values according to columns
        noise = gen.normal(0, noise_level * column_range, size = len(X_noised[column]))
        X_noised[column] = X_noised[column] + noise

    print(f"Noise added to {len(X_noised.columns)} ")

    return X_noised

# Apply blur to an image
def apply_blur_img(image, blur):
    return image.filter(ImageFilter.GaussianBlur(radius = blur))

# Apply brightness change to an image
def apply_brightness_img(image, brightness):
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(1 + brightness)



