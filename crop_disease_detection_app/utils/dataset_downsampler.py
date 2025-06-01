import os
import shutil
import random

def downsample_dataset(source_dir='PlantVillage', target_dir='sampled_dataset', num_images_per_class=10):
    os.makedirs(target_dir, exist_ok=True)
    for class_folder in os.listdir(source_dir):
        class_path = os.path.join(source_dir, class_folder)
        if os.path.isdir(class_path):
            images = os.listdir(class_path)
            selected = random.sample(images, min(len(images), num_images_per_class))
            new_class_path = os.path.join(target_dir, class_folder)
            os.makedirs(new_class_path, exist_ok=True)
            for img in selected:
                shutil.copy2(os.path.join(class_path, img), os.path.join(new_class_path, img))

if __name__ == "__main__":
    downsample_dataset()