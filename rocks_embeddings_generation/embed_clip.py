import torch
from PIL import Image
import open_clip
import os
import numpy as np
from pathlib import Path
import scipy.io as sio
import argparse

# Initialize CLIP model

def run_model(model_name, pretrained_dataset, images_folder):

    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained_dataset)
    model.eval()

    # Get all .jpg files and sort them by name
    image_files = sorted([f for f in os.listdir(images_folder) if f.lower().endswith('.jpg')])

    print(f"Found {len(image_files)} images")

    # Batch processing parameters
    batch_size = 32  # Adjust based on your GPU memory

    # Lists to store embeddings and filenames
    all_embeddings = []
    all_filenames = []

    # Process images in batches
    for i in range(0, len(image_files), batch_size):
        batch_files = image_files[i:i + batch_size]
        batch_images = []
        batch_filenames = []

        # Load and preprocess batch of images
        for filename in batch_files:
            image_path = os.path.join(images_folder, filename)

            try:
                # Load and preprocess image
                image = preprocess(Image.open(image_path))
                batch_images.append(image)
                batch_filenames.append(filename)

            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

        if batch_images:
            # Stack images into a batch tensor
            batch_tensor = torch.stack(batch_images)

            # Extract features for the entire batch
            with torch.no_grad(), torch.autocast("cuda"):
                batch_features = model.encode_image(batch_tensor)

            print(f"{batch_features.cpu().numpy().shape=}")
            all_embeddings.append(batch_features.cpu().numpy())
            all_filenames.extend(batch_filenames)

            print(f"Processed batch {i//batch_size + 1}/{(len(image_files) + batch_size - 1)//batch_size}: {len(batch_images)} images")

    # Concatenate all batches
    if all_embeddings:
        # print(f"{np.shape(all_embeddings)=}")
        embeddings = np.vstack(all_embeddings)
        print(f"{np.shape(embeddings)=}")
        filenames = all_filenames

        print(f"\nEmbeddings shape: {embeddings.shape}")

        # 1. MATLAB .mat format
        matlab_data = {
            'embeddings': embeddings,
            'filenames': np.array(filenames, dtype=object),
            'image_count': len(filenames),
            'embedding_dim': embeddings.shape[1]
        }
        sio.savemat(f'{model_name}_clip.mat', matlab_data)

        # 2. CSV format for embeddings (can be read by MATLAB)
        np.savetxt(f'{model_name}_clip.csv', embeddings, delimiter=',')

        print("Embeddings saved successfully!")
        print(f"- image_embeddings.mat: MATLAB format with embeddings and filenames")
        print(f"- image_embeddings.csv: embeddings as CSV ({embeddings.shape[0]} x {embeddings.shape[1]})")

    else:
        print("No images were successfully processed!")

# Example of how to load the embeddings in different environments
    """

    MATLAB:
    # Load .mat file:
    data = load('image_embeddings.mat');
    embeddings = data.embeddings;
    filenames = data.filenames;

    # Or load CSV and text files:
    embeddings = csvread('image_embeddings.csv');
    """

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Extract CLIP embeddings from images')
    parser.add_argument('--image_folder', type=str, required=True, help='Path to folder containing images')
    args = parser.parse_args()

    if not os.path.exists(args.image_folder):
        raise ValueError(f"Image folder does not exist: {args.image_folder}")
    


    list_models = [('RN50', 'openai'), 
                   ('RN101', 'openai'), 
                   ('ViT-B-32', 'openai'), 
                   ('ViT-L-14', 'openai'),
                   ('ViT-H-14-378-quickgelu', 'dfn5b'),
                   ('EVA02-E-14-plus', 'laion2b_s9b_b144k')]

    for i in list_models:
        print(i)
        run_model(i[0], i[1], args.image_folder)
