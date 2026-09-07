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

from timeseries_augmentations.aug.base import *
from timeseries_augmentations.aug.compose import *



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




def setup_transforms(transform_configs):
    transforms = [hy_utils.instantiate(t) for t in transform_configs.values()]
    
    for t in transforms:
        print(f"Transform: {t}, type: {type(t)}, is subclass: {isinstance(t, TimeSeriesTransform)}")

    if not transforms:
        raise ValueError("No transforms defined")
    if len(transforms) == 1:
        return transforms[0]
    return Compose(transforms)










# Function moved outside main() to avoid multiprocessing issues
def generate_image(args):
    class_label, resampling_aug, view1_aug, view2_aug, save_dir, cfg = args
    print(f"Processing class {class_label} with resampling, view1, and view2...")

    # Select a random sample from the class
    sample_index = cfg.get("selected_sample_index", 0)
    sample = class_samples[class_label][sample_index] # '''''
    x = np.array(sample["x"], dtype=np.float32)
    y = sample["y"]

    x_torch = torch.tensor(x) if not isinstance(x, torch.Tensor) else x

    # Apply augmentations
    x_resampled = resampling_aug(x_torch)
    x_view1 = view1_aug(x_torch)
    x_view2 = view2_aug(x_torch)

    # Define plot parameters
    time_steps = np.arange(x.shape[1])
    bands_to_plot = [9]

    plt.figure(figsize=(14, 6))
    for band in bands_to_plot:
        plt.plot(time_steps, x[0, :, band], label=f'Original ', linestyle='dashed')
        plt.plot(np.arange(x_resampled.shape[1]), x_resampled[0, :, band], label=f'Resampling ')
        plt.plot(np.arange(x_view1.shape[1]), x_view1[0, :, band], label=f'view1: {view1_aug}  ')
        plt.plot(np.arange(x_view2.shape[1]), x_view2[0, :, band], label=f'view2: {view2_aug} ')

        # plt.plot(time_steps, x[0, :, band], label=f'Original - Band {band}', linestyle='dashed')
        # plt.plot(np.arange(x_resampled.shape[1]), x_resampled[0, :, band], label=f'Resampling - Band {band}')
        # plt.plot(np.arange(x_view1.shape[1]), x_view1[0, :, band], label=f'{view1_aug} - Band {band}')
        # plt.plot(np.arange(x_view2.shape[1]), x_view2[0, :, band], label=f'{view2_aug} - Band {band}')

    plt.xlabel("Time (in 5-day steps)")
    plt.ylabel("Spectral value")
    plt.title(f"Augmentations - Class {y}")
    plt.legend()
    plt.grid(True)

    img_name = f"augmented_class{y}.png"
    plt.savefig(os.path.join(save_dir, img_name))
    plt.close()



@hydra.main(config_path="../config", config_name="newconfig", version_base="1.3")
def main(cfg: DictConfig):
    print("Initializing environment...")

    # random.seed(42)
    # np.random.seed(42)
    # torch.manual_seed(42)

    # Select augmentation method based on user input
    method = cfg.augmentation  # e.g., "jittering", "time_warping", etc.
    if method not in cfg.augmentations:
        raise ValueError(f"Unknown augmentation method: {method}")


    # Instantiate all models
    resampling_aug = hy_utils.instantiate(cfg.resampling)
    view1_aug = setup_transforms(cfg.augmentations[method].view1.transforms)
    view2_aug = setup_transforms(cfg.augmentations[method].view2.transforms)

    print(f"=======> {view1_aug}")

    # Select classes
    if "selected_classes" in cfg:
        selected_classes = cfg.selected_classes
        print(f"Selected classes from config: {selected_classes}")
    else:
        unique_classes = sorted(class_samples.keys())
        selected_classes = unique_classes[:5]
        print(f"No selected_classes in config. Defaulting to first 5 classes: {selected_classes}")


    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"./img_views/{method}_views/exp_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)

    # Log config
    log_path = os.path.join(save_dir, "config.txt")
    with open(log_path, "w") as log_file:
        log_file.write("Used Configuration:\n")
        log_file.write(OmegaConf.to_yaml(cfg))
        log_file.write(f"\nSelected classes: {selected_classes}\n")

    args = [(label, resampling_aug, view1_aug, view2_aug, save_dir, cfg) for label in selected_classes]

    with Pool(processes=5) as pool:
        pool.map(generate_image, args)

    print(f"All images have been saved in {save_dir}")



if __name__ == "__main__":
    main()
















# import sys
# import os
# import random
# import datetime
# import numpy as np
# import matplotlib.pyplot as plt
# import hydra
# from omegaconf import DictConfig, OmegaConf
# import hydra.utils as hy_utils
# from datasets import load_dataset
# import torch
# from tqdm import tqdm
# from multiprocessing import Pool

# # Add the project path to avoid import errors
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# # Load the dataset once
# print("Loading the dataset...")
# dataset = load_dataset("saget-antoine/francecrops", split="train")
# print("Dataset successfully loaded!")

# dataset = dataset.select(range(100))  # Reduce the dataset for faster testing

# # Organize samples into a dictionary (quick access by class)
# print("Sorting samples by class...")
# class_samples = {}
# for sample in tqdm(dataset, desc="Organizing samples", unit="sample"):
#     y = sample["y"]
#     if y not in class_samples:
#         class_samples[y] = []
#     class_samples[y].append(sample)

# print("Sample sorting completed!")


# # Function moved outside main() to avoid multiprocessing issues
# def generate_image(args):
#     class_label, resampling_aug, view1_aug, view2_aug, save_dir = args
#     print(f"Processing class {class_label} with resampling, view1, and view2...")

#     # Select a random sample from the class
#     sample = random.choice(class_samples[class_label])
#     x = np.array(sample["x"], dtype=np.float32)
#     y = sample["y"]

#     x_torch = torch.tensor(x) if not isinstance(x, torch.Tensor) else x

#     # Apply augmentations
#     x_resampled = resampling_aug(x_torch)
#     x_view1 = view1_aug(x_torch)
#     x_view2 = view2_aug(x_torch)

#     # Define plot parameters
#     time_steps = np.arange(x.shape[1])
#     bands_to_plot = [9]

#     plt.figure(figsize=(14, 6))
#     for band in bands_to_plot:
#         plt.plot(time_steps, x[0, :, band], label=f'Original - Band {band}', linestyle='dashed')
#         plt.plot(np.arange(x_resampled.shape[1]), x_resampled[0, :, band], label=f'Resampling - Band {band}')
#         plt.plot(np.arange(x_view1.shape[1]), x_view1[0, :, band], label=f'View1 - Band {band}')
#         plt.plot(np.arange(x_view2.shape[1]), x_view2[0, :, band], label=f'View2 - Band {band}')

#     plt.xlabel("Time (in 5-day steps)")
#     plt.ylabel("Spectral value")
#     plt.title(f"Augmentations - Class {y}")
#     plt.legend()
#     plt.grid(True)

#     img_name = f"augmented_class{y}.png"
#     plt.savefig(os.path.join(save_dir, img_name))
#     plt.close()



# @hydra.main(config_path="../config", config_name="newconfig", version_base="1.3")
# def main(cfg: DictConfig):
#     print("Initializing environment...")

#     random.seed(42)
#     np.random.seed(42)
#     torch.manual_seed(42)

#     # Select augmentation method based on user input
#     method = cfg.augmentation  # e.g., "jittering", "time_warping", etc.
#     if method not in cfg.augmentations:
#         raise ValueError(f"Unknown augmentation method: {method}")


#     # Instantiate all models
#     resampling_aug = hy_utils.instantiate(cfg.resampling)
#     view1_aug = hy_utils.instantiate(cfg.augmentations[method].view1.transforms)
#     view2_aug = hy_utils.instantiate(cfg.augmentations[method].view2.transforms)




#     unique_classes = list(class_samples.keys())
#     selected_classes = random.sample(unique_classes, 5)
#     print(f"Selected classes: {selected_classes}")

#     timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#     save_dir = f"./img/{method}_views/exp_{timestamp}"
#     os.makedirs(save_dir, exist_ok=True)

#     # Log config
#     log_path = os.path.join(save_dir, "config.txt")
#     with open(log_path, "w") as log_file:
#         log_file.write("Used Configuration:\n")
#         log_file.write(OmegaConf.to_yaml(cfg))
#         log_file.write(f"\nSelected classes: {selected_classes}\n")

#     args = [(label, resampling_aug, view1_aug, view2_aug, save_dir) for label in selected_classes]
#     with Pool(processes=5) as pool:
#         pool.map(generate_image, args)

#     print(f"All images have been saved in {save_dir}")



# if __name__ == "__main__":
#     main()
