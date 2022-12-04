This is the repository for our submission "**SPDET: Edge-Aware Self-Supervised Panoramic Depth Estimation Transformer With Spherical Geometry**". This repository currently only contains the model and demonstration, and we will release the full codes if the submission is accepted.

### Requirements
```
pytorch==1.11.0
cudatoolkit=10.2
numpy==1.21.5
opencv-python==4.6.0.66
timm==0.6.7
```

### Usage
1. Download our model parameters in this [link](https://drive.google.com/drive/folders/1Oh0RW_vFOY9pZKPWz0lipeRElruqFCq9?usp=sharing) and move them to the folder `checkpoints`.
2. Run the following command to test
   ```
    python test.py --checkpoints ./checkpoints/spdet-3d60.pt --image ./images/3d60.png --savedir ./results
   ```
    and the prediction results of the model are stored in the folder `results`.