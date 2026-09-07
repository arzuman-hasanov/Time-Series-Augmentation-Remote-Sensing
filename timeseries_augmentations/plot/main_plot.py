import sys
import os
import random
import datetime
import numpy as np
import matplotlib.pyplot as plt
import hydra
from omegaconf import DictConfig, OmegaConf
import hydra.utils as hy_utils
from datasets import load_dataset
import torch
from tqdm import tqdm
from multiprocessing import Pool

# Add the project path to avoid import errors
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Load the dataset once
print("Loading the dataset...")
dataset = load_dataset("saget-antoine/francecrops", split="train")
print("Dataset successfully loaded!")

dataset = dataset.select(range(100))  # Reduce the dataset for faster testing

# Organize samples into a dictionary (quick access by class)
print("Sorting samples by class...")
class_samples = {}
for sample in tqdm(dataset, desc="Organizing samples", unit="sample"):
    y = sample["y"]
    if y not in class_samples:
        class_samples[y] = []
    class_samples[y].append(sample)

print("Sample sorting completed!")


# Function moved outside main() to avoid multiprocessing issues
def generate_image(args):
    class_label, augmentation_model, save_dir, method = args
    print(f"Processing class {class_label} using {method} augmentation...")

    # Select a random sample from the class
    sample = random.choice(class_samples[class_label])
    x = np.array(sample["x"], dtype=np.float32)
    y = sample["y"]

    # Convert to a Torch tensor if necessary
    x_torch = torch.tensor(x) if not isinstance(x, torch.Tensor) else x

    # Apply augmentation
    x_augmented = augmentation_model.forward(x_torch)

    # Define plot parameters
    time_steps = np.arange(x.shape[1])
    bands_to_plot = [9]

    # Create the plot
    plt.figure(figsize=(12, 6))

    # Plot the original and augmented time series
    for band in bands_to_plot:
        plt.plot(time_steps, x[0, :, band], label=f'Original - Band {band}', linestyle='dashed')
        plt.plot(time_steps, x_augmented[0, :, band], label=f'{method} - Band {band}')

    # Customize the plot
    plt.xlabel("Time (in 5-day steps)")
    plt.ylabel("Spectral value")
    plt.title(f"Time Series with {method} - Class {y}")
    plt.legend()
    plt.grid(True)

    # Save the image
    img_name = f"{method}_class{y}.png"
    plt.savefig(os.path.join(save_dir, img_name))
    plt.close()


@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    print("Initializing environment...")

    # Set random seed
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # Select augmentation method based on user input
    method = cfg.augmentation  # e.g., "jittering", "time_warping", etc.
    if method not in cfg.augmentations:
        raise ValueError(f"Unknown augmentation method: {method}")

    print(f"Instantiating {method} model...")
    augmentation_model = hy_utils.instantiate(cfg.augmentations[method])
    print(f"{method} model loaded!")

    # Select 5 random classes
    unique_classes = list(class_samples.keys())  # Select only available classes
    selected_classes = random.sample(unique_classes, 5)
    print(f"Selected classes: {selected_classes}")

    # Create a unique folder to store generated images
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"./img/{method}/exp_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)

    # Open a log file to save the used parameters
    log_path = os.path.join(save_dir, "config.txt")
    with open(log_path, "w") as log_file:
        log_file.write("Used Configuration:\n")
        log_file.write(OmegaConf.to_yaml(cfg))
        log_file.write(f"\nSelected classes: {selected_classes}\n")

    print(f"Results saved in: {save_dir}")

    # Execute in parallel using multiprocessing (sending parameters as a tuple)
    args = [(class_label, augmentation_model, save_dir, method) for class_label in selected_classes]
    with Pool(processes=5) as pool:
        pool.map(generate_image, args)

    print(f"All images have been saved in {save_dir}")


if __name__ == "__main__":
    main()
