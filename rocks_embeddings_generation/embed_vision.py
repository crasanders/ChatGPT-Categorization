import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np
from pathlib import Path
import scipy.io as sio
import timm
import argparse

def run_model(model_name, images_folder):
    # Load model and get its preprocessing
    if model_name == 'RN50':
        weights = models.ResNet50_Weights.IMAGENET1K_V1
        model = models.resnet50(weights=weights)
        preprocess = weights.transforms()
        model.fc = nn.Identity()
        
    elif model_name == 'RN101':
        weights = models.ResNet101_Weights.IMAGENET1K_V1
        model = models.resnet101(weights=weights)
        preprocess = weights.transforms()
        model.fc = nn.Identity() 

    elif model_name == 'ViT-B-32':
        weights = models.ViT_B_32_Weights.IMAGENET1K_V1
        model = models.vit_b_32(weights=weights)
        preprocess = weights.transforms()
        model.heads = nn.Identity()
    
    elif model_name == 'ViT-B-16':
        weights = models.ViT_B_16_Weights.IMAGENET1K_V1
        model = models.vit_b_16(weights=weights)
        preprocess = weights.transforms()
        model.heads = nn.Identity()
        
    elif model_name == 'ViT-L-16':
        weights = models.ViT_L_16_Weights.IMAGENET1K_V1
        model = models.vit_l_16(weights=weights)
        preprocess = weights.transforms()
        model.heads = nn.Identity()
    
    elif model_name == 'ViT-H-14':
        weights = models.ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1
        model = models.vit_h_14(weights=weights)
        preprocess = weights.transforms()
        model.heads = nn.Identity()
    
    model.eval()
    
    # Move model to GPU if available
    device = 'cpu' #'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    print(f"Using device: {device}")
    
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
                image = Image.open(image_path)#.convert('RGB')
                image_tensor = preprocess(image)
                batch_images.append(image_tensor)
                batch_filenames.append(filename)
                
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
        
        if batch_images:
            # Stack images into a batch tensor
            batch_tensor = torch.stack(batch_images).to(device)
            
            # Extract features for the entire batch
            with torch.no_grad():
                print(f"{batch_tensor.shape=}")
                batch_features = model(batch_tensor)
                
                # Flatten if needed
                if len(batch_features.shape) > 2:
                    batch_features = batch_features.flatten(1)
            
            print(f"{batch_features.cpu().numpy().shape=}")
            all_embeddings.append(batch_features.cpu().numpy())
            all_filenames.extend(batch_filenames)
            
            print(f"Processed batch {i//batch_size + 1}/{(len(image_files) + batch_size - 1)//batch_size}: {len(batch_images)} images")
    
    # Concatenate all batches
    if all_embeddings:
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
        sio.savemat(f'{model_name}_torchvision.mat', matlab_data)
        
        # 2. CSV format for embeddings (can be read by MATLAB)
        np.savetxt(f'{model_name}_torchvision.csv', embeddings, delimiter=',')
        
        # 3. PyTorch format (optional)
        #torch.save({'embeddings': embeddings, 'filenames': filenames}, f'{model_name}_torchvision.pt')
        
        print("Embeddings saved successfully!")
        print(f"- {model_name}_torchvision.mat: MATLAB format with embeddings and filenames")
        print(f"- {model_name}_torchvision.csv: embeddings as CSV ({embeddings.shape[0]} x {embeddings.shape[1]})")
        #print(f"- {model_name}_torchvision.pt: PyTorch format")
        
    else:
        print("No images were successfully processed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract torchvision model embeddings from images')
    parser.add_argument('--image_folder', type=str, required=True, help='Path to folder containing images')
    args = parser.parse_args()

    if not os.path.exists(args.image_folder):
        raise ValueError(f"Image folder does not exist: {args.image_folder}")

    list_models = ['RN50', 'RN101', 'ViT-B-32', 'ViT-B-16', 'ViT-L-16', 'ViT-H-14']
    
    for model_name in list_models:
        print(f"\n{'='*60}")
        print(f"Processing with {model_name}")
        print(f"{'='*60}")
        run_model(model_name, args.image_folder)
