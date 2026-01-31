Install the following packages - 

```bash
pip install timm
pip install open_clip_torch
pip install scipy
```

To extract CLIP embeddings - 

```bash
python embed_clip.py --image_folder="PATH_TO_360_ROCK_IMAGES"
```

To extract embeddings from (torch)vision models - 

```bash
python embed_vision.py --image_folder="PATH_TO_360_ROCK_IMAGES"
```

The image_embeddings.zip contains the generated embeddings from all the models in .csv and .mat format.