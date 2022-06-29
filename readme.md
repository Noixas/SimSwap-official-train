To run this code in June 2022 follow the next steps:

1. Clone repo 
2. Manually copy big files that aren't in the repo:
    - `arcface_model/arcface_checkpoint.tar`
    - `insightface_func/models/antelope/scrfd_10g_bnkps.onnx`
    - `insightface_func/models/antelope/glintr100.onnx`
    - Copy train dataset into fast storage (SSD. (In my pc it is located at `/home/astro/Documents/UvA/Thesis/vggface2_crop_arcfacealign_224`)
3. Install conda environment
    - Python 3.8
    - Pytorch 1.11.0 and Cuda 11.6 works
    - pip install timm 
    - pip install wandb
    
