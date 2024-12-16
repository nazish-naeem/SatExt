# Introduction 
Satellite imaging is pivotal in enabling remote sensing and monitoring of our planet. As a result, it can support many applications such as climate monitoring, natural resource management, agricultural monitoring, defense, etc. Unfortunately, despite the presence of more than 1k satellites in space, today we cannot perform frequent remote monitoring of a specific region. This is because the current remote sensing algorithms rely on only one type of satellite. Therefore its operation is limited by the frequency at which that specific satellite visits a certain location. Additionally, the presence of clouds, malfunction in the sensors, and corrupted data further limit the frequency of monitoring the Earth.

This work proposes, SatExt, which is a spectral, spatial, and temporal extension of the satellite images for remote monitoring tasks. The primary aim is to leverage generative AI (GenAI) to standardize satellite images into a unified format—specifically, the Sentinel-2 format. This choice is motivated by Sentinel-2's high dimensionality in both spectral and spatial domains, as well as its widespread use as a baseline for most remote sensing algorithms. 

The first step, termed Spectral Extension, leverages the spectral information available in the input satellite images (e.g., Landsat) to reconstruct the Sentinel-2 frequency spectrum while preserving the original spatial resolution. The second step focuses on enhancing spatial resolution. Here, a Diffusion Model ,inspired by [1], is employed to transform low-resolution data into high-resolution imagery, effectively generating fine-grained details from noise. Here we show an end-to-end SatExt system flow:

<img width="1103" alt="SatExt-e2e" src="https://github.com/user-attachments/assets/592fbc8f-4399-4e50-ac64-f08126f37616" />

Following we show an animation of the Diffusion model in action, when generating high spatial resolution Sentinel-2 images from noise given low resolution images.


https://github.com/user-attachments/assets/7d826396-7bb1-40ab-9e5c-ec8177ec64ca



The use of a generative AI allows us to extend this project in the future to cloud inpainting, fixing corrupted data, multi-modal extension for generating dummy data, etc.

<table>
  <tr>
    <td align="center">Extending to Multiple Satellite</td>
     <td align="center">Cloud Inpainting</td>
  </tr>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/e3f82cab-63c5-4400-bc65-42fad24a4e3e" width=500></td>
    <td><img src="https://github.com/user-attachments/assets/8f58707d-734e-41b2-a692-911eee206571" width=500></td>
  </tr>

  <tr>
    <td align="center">Multimodal Extension using Stable Diffusion</td>
     <td align="center">Fixing Corrupted Sensor Data </td>
  </tr>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/fcae1cc6-45e3-421f-832f-08f121740d89" width=500></td>
    <td><img src="https://github.com/user-attachments/assets/7445ee56-4346-49a6-87db-c17ffb5c36b2" width=500></td>
  </tr>
 </table>




# Getting Started
Clone the github repo
```
git clone
cd RemoteSensingFoundationModels
```
create python environment and install the requirements
```
python3 -m venv env
source env/bin/activate
pip install -r Spectral-Extension/requirements.txt
```
## Data Generation

We generated data using [Planetary Computer](https://planetarycomputer.microsoft.com/). The dataset can be generated by using the following commands

```
cp farmvibes-download/dynamic_world_samples.geojson Spectral-Extension/gen_data
cd Spectral-Extension/gen_data
python make_medium_dataset.py
```
Make sure that the generated images are in the right folder and the name of the folder is correct before using them.


```bash
├── path/to/data
│   ├── Landsat
|       ├── [LS_file_name]_dw_sample_num.tif
│   ├── S2_aligned
|       ├── [S2_file_name]_dw_sample_num.tif
```


# Spectral Extension

### Training
To train the spectral extension model, run the following command
```
cd Spectral-Extension
python train_ae.py --resume_path [path/to/model/resume/the/training/from, None otherwise] --model [type of model, choose from 'RRDB','RCAN','FCONV','AE'] --lr [learning rate]
```

This module allows us to choose the model from Residual in Residual Dense Block(RRDB), Residual Channel Attenuation Network(RCAN)[2], Fully Convolutional Netwrok(FCONV), and Autoencoder(AE). This file saves the results/checkpoints in folder Spectral-Extension/experiments

Note that here you do not have to give the path of the validation data. Given the path of the dataset the dataloader performs an 80-20 split with a seed so everytime the dataloader is called the split remains unchanged.



### Testing
run the following commands to perform evaluation
```
python eval.py --resume_path [path/to/model/resume/the/training/from, None otherwise] --model [type of model, choose from 'RRDB','RCAN','FCONV','AE']
```
the final results are saved in Spectral-Extension/eval/.
`[model name].txt` saves the mean and variance of psnr and ssim of all the sentinel-2 bands


# Super Resolution

### Training 

To train the super resolution model run the following command
```
cd Image-Super-Resolution-Iterative-Refinement
python sr_our.py -c config/sr_S2_RGB.json
```
the config file contains all the information about the test and validation data, model parameters, noise scheduler, testing and training settings. All the hyper-parameters can be changed by changing the coonfig file.
The checkpoints are saved in `Image-Super-Resolution-Iterative-Refinement/experiments`
Note that super resolution only needs Sentinel-2 images for training and testing, therefore the dataset path should be `path/to/data/S2_aligned`
### Inference 
to run the inference on the testing data, run the following command
```
python infr.py -c config/sr_S2_RGB_infr.json
```




# Run End-to-End Test
After training both steps, end-to-end system can be run using the following commands

```
python main.py --config [path/to/super-resolution/config/file] --resume_path [path/to/spectral-extension/model/weights] --model [the type of spectral extension model in use. Choose from 'RCAN', 'FCONV', 'RRDB'] --data_path [path/to/data]
```
the results are saved under `experiments`.


# References

[1] Saharia, Chitwan, et al. "Image super-resolution via iterative refinement." IEEE transactions on pattern analysis and machine intelligence 45.4 (2022): 4713-4726.
[2] Yulun Zhang, Kunpeng Li, Kai Li, Lichen Wang, Bineng Zhong, and Yun Fu, “Image super-resolution using very deep residual channel attention networks,” in Proceedings of the European conference on computer vision (ECCV), 2018, pp. 286–301.

Credits: The Super Resolution code is derived from [here](https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement). Which is an unoffical implementeation of [1]. You can find the original readme in the `Image-Super-Resolution-Iterative-Refinement` to get a better understanding of how it works.

This work was a part of a three month summer internship at Microsoft Research mentored by Peder Olsen and Vaishnavi Ranganathan.

