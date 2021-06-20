# Bridging the Visual Gap: Wide-Range Image Blending
PyTorch implementaton of our CVPR 2021 oral paper "Bridging the Visual Gap: Wide-Range Image Blending".  
You can visit our project website [here](https://julia0607.github.io/Wide-Range-Image-Blending/).

In this paper, we propose a novel model to tackle the problem of wide-range image blending, which aims to smoothly merge two different images into a panorama by generating novel image content for the intermediate region between them.
<div align=center><img height="230" src="https://github.com/julia0607/Wide-Range-Image-Blending/blob/main/samples/teaser.gif"/></div>

## Paper
[Bridging the Visual Gap: Wide-Range Image Blending](https://arxiv.org/abs/2103.15149)  
[Chia-Ni Lu](mailto:julialu67.cs08g@nctu.edu.tw), [Ya-Chu Chang](mailto:jenna.cs07g@nctu.edu.tw), [Wei-Chen Chiu](https://walonchiu.github.io/)  
IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2021.

Please cite our paper if you find it useful for your research.  
```
@InProceedings{lu2021bridging,
    author = {Lu, Chia-Ni and Chang, Ya-Chu and Chiu, Wei-Chen},
    title = {Bridging the Visual Gap: Wide-Range Image Blending},
    booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {June},
    year = {2021}
}
```

## Installation
* This code was developed with Python 3.7.4 & Pytorch 1.0.0 & CUDA 9.2
* Other requirements: numpy, skimage, tensorboardX
* Clone this repo
```
git clone https://github.com/julia0607/Wide-Range-Image-Blending.git
cd Wide-Range-Image-Blending
```

## Testing
Download our pre-trained model weights from [here](https://drive.google.com/drive/folders/1YNN6_rhNXlOkunXZ0Ynj7SkObtTemHDY?usp=sharing) and put them under `weights/`. 

Test the sample data provided in this repo:
```
python test.py
```
Or download our paired test data from [here](https://drive.google.com/file/d/1G6mqSqx3XtVAsGWBnyxnAtSaocSIETkm/view?usp=sharing) and put them under `data/`.  
Then run the testing code:
```
python test.py --test_data_dir_1 ./data/scenery6000_paired/test/input1/
               --test_data_dir_2 ./data/scenery6000_paired/test/input2/
```

Run your own data:
```
python test.py --test_data_dir_1 YOUR_DATA_PATH_1
               --test_data_dir_2 YOUR_DATA_PATH_2
               --save_dir YOUR_SAVE_PATH
```
If your test data isn't paired already, add `--rand_pair True` to randomly pair the data.

## Training
We adopt the scenery dataset proposed by [Very Long Natural Scenery Image Prediction by Outpainting](https://github.com/z-x-yang/NS-Outpainting) for conducting our experiments, in which we split the dataset to 5040 training images and 1000 testing images. 

Download the dataset with our split of train and test set from [here](https://drive.google.com/file/d/1TLh2Gg_iLf3BR3EcqJ0BTh17U6yCq2dD/view?usp=sharing) and put them under `data/`.   
You can unzip the `.zip` file with `jar xvf scenery6000_split.zip`.   
Then run the training code for self-reconstruction stage (first stage):
```
python train_SR.py
```
After finishing the training of self-reconstruction stage, move the latest model weights from `checkpoints/SR_Stage/` to `weights/`, and run the training code for fine-tuning stage (second stage):
```
python train_FT.py --load_pretrain True
```

Train the model with your own dataset:
```
python train_SR.py --train_data_dir YOUR_DATA_PATH
```
After finishing the training of self-reconstruction stage, move the latest model weights to `weights/`, and run the training code for fine-tuning stage (second stage):
```
python train_FT.py --load_pretrain True
                   --train_data_dir YOUR_DATA_PATH
```
If your train data isn't paired already, add `--rand_pair True` to randomly pair the data in the fine-tuning stage.

## TensorBoard Visualization
Visualization on TensorBoard for training and validation is supported. Run `tensorboard --logdir YOUR_LOG_DIR` to view training progress.

## Acknowledgments
Our code is partially based on [Very Long Natural Scenery Image Prediction by Outpainting](https://github.com/z-x-yang/NS-Outpainting) and a pytorch re-implementation for [Generative Image Inpainting with Contextual Attention](https://github.com/daa233/generative-inpainting-pytorch).  
The implementation of ID-MRF loss is borrowed from [Image Inpainting via Generative Multi-column Convolutional Neural Networks](https://github.com/shepnerd/inpainting_gmcnn/tree/ba7f7109c38c3805800283cdb9d79cd7c4a3294f).