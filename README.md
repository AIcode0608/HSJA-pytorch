# HSJA-pytorch
This repository contains the PyTorch implementation of HSJA, capable of reproducing results on the CIFAR10, SVHN, and Caltech256 datasets as reported in the paper.

We introduce a universal hard-label black-box model theft method: HSJA. Initially, HSJA uses Class Activation Mapping (CAM) to analyze sample attributes and extract sensitive feature regions, thereby reducing the noise impact on theft training and enhancing the precision of model functionality theft. Furthermore, we've constructed an efficient active learning selection strategy to train a substitute model with higher similarity within the same query budget. Ultimately, we propose a joint enhancement training mechanism, enabling the substitute model to approximate the decision distribution of the target black-box more effectively with fewer samples, addressing the balance between query budget and theft accuracy. We've validated the efficacy of HSJA on CIFAR10, SVHN, CALTECH256 datasets, and real APIs. Notably, using consistent experimental parameters, our substitute model's consistency accuracy has improved by up to 4.56% compared to existing methods.

<img src="https://github.com/AIcode0608/HSJA-pytorch/blob/main/assets/pig2.png">

## Requirements
- python 3.6 +
- pytorch 1.6.0
- torchvision 0.7.0
- tensorboard 2.3.0
- pillow

## Install dependencies
Install the required dependencies with the following command:
```
pip install -r requirements.txt
```
If you encounter pip version issues, try updating pip and reinstalling:
```
pip install --upgrade pip
```

## Prepare the train data

The objective of this project is to secure high-quality training data by downloading and processing images from the ILSVRC-2012 dataset and utilizing a pre-trained Imagenet model for high-confidence filtering. Here are the detailed steps:
* Download the ILSVRC-2012 Dataset: Ensure you've downloaded the dataset with 1.2 million images. Secure it from a reliable source and save it locally.
* Import Necessary Libraries:
```
import torch
import torchvision.models as models
```
* Load the Pre-trained Model: We're using ResNet-50 for demonstration, but you can choose other models if necessary:Load the Pre-trained Model: We're using ResNet-50 for demonstration, but you can choose other models if necessary:
```
model = models.resnet50(pretrained=True)
```
* Image Preprocessing:
```
* preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```
* Set the Model to Evaluation Mode：
```
model.eval()
```
* Dataset Filtering and Selection: After the above steps, you can perform inference on the entire dataset with the pre-trained model and select high-confidence images. For diversity, uniformly select samples from each Imagenet category. We sort each category by confidence and select the top k samples. Compute k as 122,186 (number of categories in ILSVRC-2012) divided by the Imagenet dataset categories used.
* H-SNR Feature Extraction: To optimize the dataset further, we use CAM to extract sensitive areas. This process leverages the Signal-to-Noise Ratio (SNR) metric to quantify the relative strength of primary and background features. Find our high SNR feature extraction algorithm's code at the provided link.

* Code File Structure:After completing dataset filtering and selection, you can organize the relevant code into a single file for easy replacement of the dataset path and project execution. Our approach is to store the dataset categorized in different folders and in .jpg format.

## How to Use to Train

To train the model, follow the steps below:
* Clone the repository：Use Git or other version control tools to clone the repository to your local computer:
```
* git clone <repository_url>
```
* Python version: Make sure your Python version is 3.x or above, as Python 3+ is currently supported.
* Modify the dataset path: Before starting the training, please modify lines 263, 272, and 281 in the ssl_dataset.py file to match the location of your dataset on the server. This ensures that the training process can load the dataset correctly.Please note the differences in the paths of the datasets we need to modify:
```
origin_data_files: Address for step-1 initial substitute model training data.
all_data_files: Address for all unlabeled data.
```
Note: We read data in JSON format. If your dataset storage differs, please modify the data reading method in ssl_dataset.
* Prepare the Black-box Model: Before training the surrogate model, ensure the black-box model is ready. In this experiment, we use a ResNet34 model trained on CIFAR-10. When saving, save the complete model + parameters. Adjust the network structure parameters as needed during loading. The black-box model loading code is at line 54 of ssl_dataset.py.
* PS: We've trained a [CIFAR10 black-box](https://github.com/AI-See-World/ASSL-pytorch/tree/master) model, which you can directly integrate into your project.
* Set Important Hyperparameters: In the train.py main function, you can set these critical hyperparameters:

```
--save_dir : Specify the directory where the model will be saved. The default is ./saved_models.

--save_name : Specify the filename of the saved model. The default is cifar10-40.

--resume : If set to True, resume training from the previous training session. The default is False.

--load_path : Specify the path of the pretrained model to load. The default is None.

--overwrite : If set to True, overwrite the existing model with the same name when saving the model. The default is True.

--epoch : Specify the total number of training epochs. The default is 10.

--num_train_iter : Specify the total number of training iterations (equal to epoch multiplied by every_num_train_iter).

--every_num_train_iter : Specify the number of training iterations per epoch.
```
Now, you can run the following command to start the training process. Make sure you are in the root directory of the repository and execute the following command in the command line or terminal:

```
python train.py
```
## Result
The following tables showcase HSJA's excellent theft performance on various target black-boxes. Throughout the experiments, our method consistently outperforms others with a noticeable margin of higher consistency accuracy.

### CIFAR10
| Method | 10K | 15K | 20K | 25K | 30K| 
|:---|:---:|:---:|:---:|:---:|:---:|
|Knockoff|	48.64|	55.78	|63.71	|68.52	|73.81|
|ActThief(Entropy)|	41.73|	49.32|	58.19|	66.06|	73.50|
|ActThief(K-center)|	52.79|	59.40	|65.18|	70.36|	74.22|
|Dissector(K-center)|	58.91|	66.54|	74.57	|77.46	|79.36|
|Dissector(Random)|	64.53|	70.12|	76.89|	78.90|	80.12|
|HSJA(Ours)|	68.92|	74.24|	79.31|	80.41|	81.36|

<img src="https://github.com/AIcode0608/HSJA-pytorch/blob/main/assets/CIFAR10.png" width="310px"><img src="https://github.com/AIcode0608/HSJA-pytorch/blob/main/assets/SVHN.png" width="310px"><img src="https://github.com/AIcode0608/HSJA-pytorch/blob/main/assets/Caltech256.png" width="310px">


### SVHN
| Method | 10K | 15K | 20K | 25K | 30K| 
|:---|:---:|:---:|:---:|:---:|:---:|
|Knockoff|	49.98|	61.21|	73.10|	77.86|	84.57|
|ActThief(Entropy)|	74.63|	80.16|	85.71|	87.26|	89.99|
|ActThief(K-center)|	44.37|	56.24|	67.44|	74.78|	81.32|
|Dissector(K-center)|	80.01|	84.63|	87.76|	88.66|	90.09|
|Dissector(Random)|	83.42|	85.56|	88.92|	90.03|	92.68|
|HSJA(Ours)|	85.77|	87.43|	90.36|	91.47|	93.21|


### CALTECH256
| Method | 10K | 15K | 20K | 25K | 30K| 
|:---|:---:|:---:|:---:|:---:|:---:|
|Knockoff	|44.66|	47.39|	53.36|	54.25|	55.63|
|ActThief(Entropy)	|41.18	|46.82	|53.24	|53.86	|54.28|
|ActThief(K-center)	|47.88	|53.36|	56.03	|56.91|	58.23|
|Dissector(K-center)	|54.63	|56.76	|60.69	|62.81|	63.76|
|Dissector(Random)	|49.48	|53.31	|56.85|	57.09|	58.67|
|HSJA(Ours)	|56.96	|58.48	|62.14	|63.96|	65.01|


