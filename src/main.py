import cv2
import numpy as np
import requests
import os
import tempfile
import shutil
from pathlib import Path
from typing import List, Tuple
from sklearn.neighbors import KNeighborsClassifier
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from dotenv import load_dotenv

load_dotenv()
# Replace with your Unsplash API access key
UNSPLASH_API_ACCESS_KEY = os.environ.get("UNSPLASH_API_ACCESS_KEY")
def download_images(query: str, count: int) -> List[str]:
    for i in range(int(count/10)):
        url = f"https://api.unsplash.com/search/photos?query={query}&page={i+1}"
        headers = {"Authorization": f"Client-ID {UNSPLASH_API_ACCESS_KEY}"}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()

        image_urls = [img["urls"]["small"] for img in data["results"]]

        temp_dir = tempfile.mkdtemp()
        image_paths = []

        for idx, img_url in enumerate(image_urls):
            response = requests.get(img_url, stream=True)
            response.raise_for_status()

            img_path = os.path.join(temp_dir, f"{idx}.jpg")
            with open(img_path, "wb") as f:
                response.raw.decode_content = True
                shutil.copyfileobj(response.raw, f)

            image_paths.append(img_path)

    return image_paths

def load_images(image_paths: List[str], size: int) -> List[np.ndarray]:
    images = []

    for path in image_paths:
        img = cv2.imread(path)
        resized_img = resize_with_aspect_ratio(img, (size, size))
        images.append(resized_img)

    return images

def resize_with_aspect_ratio(img: np.ndarray, target_size: Tuple[int,int]) -> np.ndarray:
    height, width = img.shape[:2]
    target_width, target_height = target_size

    # Compute the scaling factor while preserving the aspect ratio
    scaling_factor = min(target_width / width, target_height / height)

    # Compute the new dimensions for the image
    new_width = int(width * scaling_factor)
    new_height = int(height * scaling_factor)

    # Resize the image with the new dimensions
    resized_img = cv2.resize(img, (new_width, new_height))

    return resized_img

def average_color(img):
    return np.average(np.average(img, axis=0), axis=0)

def preprocess_images(images):
    # Resize and normalize images
    resized_images = [cv2.resize(img, (32, 32)) for img in images]
    normalized_images = [(img / 255.0).astype(np.float32) for img in resized_images]

    return normalized_images

def train_knn(images, features):
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(features, np.arange(len(images)))
    return knn

def extract_cnn_features(model, images):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    processed_images = [transform(img).unsqueeze(0) for img in images]
    features = [model(img).detach().numpy().flatten() for img in processed_images]

    return features

def find_best_match(img, dataset, model, features, use_knn=True):
    resized_img = cv2.resize(img, (32, 32))
    normalized_img = (resized_img / 255.0).astype(np.float32)

    if use_knn:
        feature = average_color(normalized_img).reshape(1, -1)
        best_idx = model.predict(feature)[0]
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        input_img = transform(normalized_img).unsqueeze(0)
        feature = model(input_img).detach().numpy().flatten()
        best_idx = np.argmin(np.linalg.norm(features - feature, axis=1))

    return dataset[best_idx]

def create_mosaic(target_img, mosaic_dataset, cell_size, model, features, use_knn=True):
    mosaic = np.zeros_like(target_img)

    for x in range(0, target_img.shape[1], cell_size):
        for y in range(0, target_img.shape[0], cell_size):
            cell = target_img[y : y + cell_size, x : x + cell_size]
            best_match = find_best_match(cell, mosaic_dataset, model, features=features, use_knn=use_knn)
            actual_cell_height = min(cell_size, mosaic.shape[0] - y)
            actual_cell_width = min(cell_size, mosaic.shape[1] - x)
            mosaic[y : y + actual_cell_height, x : x + actual_cell_width] = cv2.resize(best_match, (actual_cell_width, actual_cell_height))
    return mosaic

def create_mosaic_from_uploaded_image(
    target_image_path: str,
    text_prompt: str,
    cell_size: int,
    model_choice: str,
    output_folder: Path,
):
    print("Downloading images...")
    image_paths = download_images(text_prompt, 100)
    mosaic_dataset = load_images(image_paths, cell_size)

    print("Preprocessing images...")
    preprocessed_images = preprocess_images(mosaic_dataset)

    print("Training kNN model...")
    knn_features = [average_color(img) for img in preprocessed_images]
    knn_model = train_knn(mosaic_dataset, knn_features)

    print("Training CNN model...")
    cnn_model = models.resnet18(pretrained=True)
    cnn_model = torch.nn.Sequential(*list(cnn_model.children())[:-1])
    cnn_model.eval()

    print("Creating mosaic...")
    target_img = cv2.imread(target_image_path)

    if model_choice == "knn":
        mosaic_img = create_mosaic(target_img, mosaic_dataset, cell_size, knn_model, features=knn_features, use_knn=True)
    elif model_choice == "cnn":
        cnn_features = extract_cnn_features(cnn_model, preprocessed_images)
        mosaic_img = create_mosaic(target_img, mosaic_dataset, cell_size, cnn_model, features=cnn_features, use_knn=False)
    else:
        raise ValueError("Invalid model choice")

    # Save the final mosaic
    output_filename = f"{Path(target_image_path).stem}_mosaic_output.jpg"
    output_path = output_folder / output_filename
    cv2.imwrite(str(output_path), mosaic_img)
    print(f"Mosaic saved to {output_path}")

    return output_filename
