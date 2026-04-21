import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Save figure
def save_figure(figure, filename, config):
    save_path = os.path.join(config['paths']['results_figures'], filename)
    figure.savefig(save_path, dpi = 150, bbox_inches ='tight')
    print(f"Saved: {save_path}")
    plt.close(figure)

# Plot class distribution bar chart
def plot_class_distribution(report, title, config):
    figure, axes = plt.subplots(figsize = (8, 5))

    class_distribution = report['class_distribution']
    classes = [str(k) for k in class_distribution.keys()]
    counts = list(class_distribution.values())

    axes.bar(classes, counts, color = sns.color_palette("Set2", len(classes)))
    axes.set_xlabel("Class")
    axes.set_ylabel("Count")
    axes.set_title(f"Class Distribution-{title}")

    for i, count in enumerate(counts):
        axes.text(i, count + max(counts) * 0.01, str(count), ha = 'center', fontsize = 9)

    save_figure(figure, f"class_distribution_{title.lower().replace(' ', '_')}.png", config)

# Plot correlation heatmap
def plot_correlation(report, title, config):
    figure, axes = plt.subplots(figsize = (10, 8))

    correlation = report['correlation']
    sns.heatmap(correlation, annot = False, cmap = 'coolwarm', center = 0, ax = axes)
    axes.set_title(f"Feature Correlation-{title}")

    save_figure(figure, f"correlation_{title.lower().replace(' ', '_')}.png", config)

# Plot feature boxplot
def plot_feature_boxplots(X, title, config):
    figure, axes = plt.subplots(figsize = (12, 6))

    X.boxplot(ax = axes, rot = 45)
    axes.set_title(f"Feature Distribution-{title}")
    axes.set_ylabel("Value")

    save_figure(figure, f"boxplot_{title.lower().replace(' ', '_')}.png", config)

# Plot accuracy comparison across models
def plot_accuracy_comparison(accuracies, model_names, title, config):
    figure, axes = plt.subplots(figsize = (8, 5))
    colors = sns.color_palette("Set2", len(model_names))
    bars = axes.bar(model_names, accuracies, color = colors)
    axes.set_ylabel("Accuracy")
    axes.set_title(f"Model Accuracy-{title}")
    axes.set_ylim(0, 1.0)

    for bar, accuracy in zip(bars, accuracies):
        axes.text(bar.get_x() + bar.get_width() / 2, accuracy + 0.01, f"{accuracy:.4f}", ha = 'center', fontsize = 9)

    save_figure(figure, f"accuracy_{title.lower().replace(' ', '_')}.png", config)

# Plot 1 - Perturbation comparison
def plot_perturbation_comparison(baseline_accuracy, perturbation_accuracies, perturbation_levels, perturbation_name, config):
    figure, axes = plt.subplots(figsize = (8, 5))
    levels = ['Baseline'] + [str(l) for l in perturbation_levels]
    accuracies = [baseline_accuracy] + perturbation_accuracies

    axes.plot(levels, accuracies, marker = 'o', linewidth = 2, color = 'red')
    axes.set_xlabel(perturbation_name)
    axes.set_ylabel('Accuracy')
    axes.set_title(f"Impact of {perturbation_name} on accuracy")
    axes.axhline(y = baseline_accuracy, linestyle = '--', color = 'gray', alpha = 0.5, label = 'Baseline')
    axes.legend()
    save_figure(figure, f"perturbation_{perturbation_name}.png", config)

# Plot 2 - Seed variation result
def plot_seed_variation(results, title, config):
    figure, axes = plt.subplots(figsize = (10, 5))
    accuracies = results['accuracies']
    seeds = range(len(accuracies))
    axes.bar(seeds, accuracies, color = sns.color_palette("Set2", len(accuracies)))
    axes.axhline(y = results['mean_accuracy'], color = 'blue', linestyle = '--', alpha = 0.5, label = f'Mean {results["mean_accuracy"]:.4f}')
    axes.set_xlabel("Seed")
    axes.set_ylabel("Accuracy")
    axes.set_title(f"Seed Variation - {title}")
    axes.legend()

    save_figure(figure, f"seed_variation_{title.lower().replace(' ', '_')}.png", config)

# Plot 3 - Data variation result
def plot_data_size_variation(results, title, config):
    figure, axes = plt.subplots(figsize = (8, 5))
    fractions = []
    for fr in results['fraction_results']:
        fractions.append(fr['fraction'])
    accuracies = []
    for fr in results['fraction_results']:
        accuracies.append(fr['accuracy'])

    axes.plot(fractions, accuracies, marker = 'o', linewidth = 2, color = 'red')
    axes.set_xlabel("Fractions of training data")
    axes.set_ylabel("Accuracy")
    axes.set_title(f"Data size impact - {title}")
    axes.set_xticks(fractions)
    new_labels = []
    for f in fractions:
        percentage = round(f * 100, 2)
        label = f"{percentage:.0f}%"
        new_labels.append(label)
    axes.set_xticklabels(new_labels)

    save_figure(figure, f"data_size_variation_{title.lower().replace(' ', '_')}.png", config)

# Plot 4 - Flagging distribution
def plot_flag_distribution(flags, title, config):
    figure, axes = plt.subplots(figsize = (6, 6))
    red = int(np.sum(flags == 'RED'))
    yellow = int(np.sum(flags == 'YELLOW'))
    green = int(np.sum(flags == 'GREEN'))
    count = [green, yellow, red]
    labels = [f"GREEN\n{green}", f"YELLOW\n{yellow}", f"RED\n{red}"]
    colors = ['green', 'yellow', 'red']
    axes.pie(count, labels = labels, colors = colors, autopct = '%1.1f%%', startangle = 90)
    axes.set_title(f"Prediction Flags - {title}")

    save_figure(figure, f"flagging_distribution_{title.lower().replace(' ', '_')}.png", config)

# Plot 5 - Uncertainty distribution
def plot_uncertainty_distribution(uncertainty, title, config):
    figure, axes = plt.subplots(figsize = (8, 5))
    axes.hist(uncertainty, bins = 50, alpha = 0.7)
    axes.axvline(x = config['stage3_inference']['flagging']['uncertainty_threshold'], linestyle = '--', color = 'red', label = 'Threshold')
    axes.set_xlabel("Uncertainty Score")
    axes.set_ylabel("Count")
    axes.set_title(f"Uncertainty Distribution - {title}")
    axes.legend()

    save_figure(figure, f"uncertainty_distribution_{title.lower().replace(' ', '_')}.png", config)

# Plot Grad-CAM heatmaps for sample images
def plot_gradcam_sample(test_set, heatmaps, predictions, num_samples, config, title, save = True):
    columns = 4
    rows = (num_samples + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns * 2, figsize = (20, 3 * rows))

    for i in range(num_samples):
        row = i // columns
        column = (i % columns) * 2

        image, label = test_set[i]
        # Convert tensor to displayable image
        image_display = image.permute(1, 2, 0).numpy()
        image_display = (image_display - image_display.min()) / (image_display.max() - image_display.min())

        # Original image
        axes[row][column].imshow(image_display)
        axes[row][column].set_title(f"[{i}] prediction:{predictions[i]}")
        axes[row][column].axis('off')

        # heatmap overlay
        axes[row][column + 1].imshow(image_display)
        axes[row][column + 1].imshow(heatmaps[i], cmap = 'jet', alpha = 0.5)
        axes[row][column + 1].set_title(f"[{i}] heatmap")
        axes[row][column + 1].axis('off')

    figure.suptitle(f"Grad-CAM samples {title}")
    figure.tight_layout()
    if save:
        save_figure(figure, f"gradcam_sample_{title.lower().replace(' ', '_')}.png", config)
    else:
        plt.show()

