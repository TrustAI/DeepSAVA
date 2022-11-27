# Sparse Adversarial Video Attack with Spatial Transformation
Ronghui Mu, Wenjie Ruan, Leandro Soriano Marcolino and Qiang Ni.

Sparse Adversarial Video Attack with Spatial Transformation

The paper is accepted by the The 32nd British Machine Vision Conference (BMVC).
https://arxiv.org/abs/2111.05468


Email: ronghui.mu@lancaster.ac.uk
# Abstract
In recent years, a significant amount of research efforts concentrated on adversarial attacks on images, while adversarial video attacks have seldom been explored. We propose an adversarial attack strategy on videos, called DeepSAVA.  Our model includes both additive perturbation and spatial transformation by a unified optimisation framework, where the structural similarity index measure is adopted to measure the adversarial distance. We design an effective and novel optimisation scheme which alternatively utilizes Bayesian optimisation to identify the most influential frame in a video and Stochastic gradient descent (SGD) based optimisation to produce both additive and spatial-transformed perturbations. Doing so enables DeepSAVA to perform a very sparse attack on videos for maintaining human imperceptibility while still achieving state-of-the-art performance in terms of both attack success rate and adversarial transferability. Our intensive experiments on various types of deep neural networks and video datasets confirm the superiority of DeepSAVA.

# Generated adversarial video samples


[![Everything Is AWESOME]](https://www.youtube.com/channel/UCBDswZC2QhBhTOMUFNLchCg)







The generated video can be found in https://www.youtube.com/channel/UCBDswZC2QhBhTOMUFNLchCg

# Results
<img width="669" alt="截屏2021-10-22 下午3 17 37" src="https://user-images.githubusercontent.com/41231651/138469948-196edeca-45d0-4268-8c21-0cab10ed5815.png">
<img width="664" alt="截屏2021-10-22 下午3 18 45" src="https://user-images.githubusercontent.com/41231651/138470150-7315272e-b960-4e07-acb2-a0d3cdf7660e.png">

# Run
The code is tested on the tensorfow >= 1.3.0
## Prepare data
UCF101 can be downloaded and extracted following the instructions in https://github.com/harvitronix/five-video-classification-methods

HMDB51 can be downloaded as RGB images in https://github.com/feichtenhofer/twostreamfusion

UCF101 data need to be  stored under UCF101/video_data/test 
HMDB51 data need to be stored under HMDB51/video_data/test
## checkpoints
The checkpoints for UCF101 LSTM+CNN can be downloaded in https://github.com/yanhui002/video_adv

Please download the checkpoints for I3D and  I3V for UCF101 and HMDB from https://www.dropbox.com/sh/o8ub94d2ecnzrgk/AACnY6-iPHgiFpPm0FX_4DkLa?dl=0


## Generate adversarial examples constrained by maximum iteration 100 for I3D model
python test_gen.py -i video_data -o output/I3D --model i3d_inception --dataset UCF101 --file_list /video_data/batch_test/test_saved.csv -- constraint iteration
## Generate adversarial examples constrained by maximum ssim loss 0.08 for I3D model 
python test_gen.py -i video_data -o output/I3D --model i3d_inception --dataset UCF101 --file_list /video_data/batch_test/test_saved.csv -- constraint ssim --budget 0.08 --num_iter 500
## Generate adversarial examples constrained by maximum lp norm 0.1 for I3D model
python test_gen.py -i video_data -o output/I3D --model i3d_inception --dataset UCF101 --file_list /video_data/batch_test/test_saved.csv --constraint lp --budget 0.1 --num_iter 500 
The generated adversarial videos will be stored in the folder "output/I3D"
## Adversarial training for inception-v3 model
python train_cnn.py -i 'Data path'
## Adversarial training for CNN-LSTM model
### First extract features use the adversarial trained inception-v3 model
python extract_features.py -i 'Data path'
### Then input the extracted features to lstm to train
python train_new.py -i 'Feature path'
## Adversarial training for I3D model
### The I3D model is based on I3D models trained on Kinetics https://github.com/deepmind/kinetics-i3d.git
python train_i3d.py -i 'Data path'


-- Ronghui Mu & Wenjie Ruan

