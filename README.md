# HUST-OBS
Oracle Bone Script data collected by VLRLab of HUST
We have open-sourced the HUST-OBS dataset and the models used in the dataset, including: Chinese OCR, MoCo, and the ResNet50 for Validation.
## HUST-OBS Dataset
[HUST-OBS Download](https://figshare.com/s/8a9c0420312d94fc01e3)
### Tree of our dataset
- HUST-OBS
  - deciphered
    - ID1
      - Source_ID1_Filename
      - Source_ID1_Filename
      - .....
    - ID2
      - Source_ID2_Filename
      - .....
    - ID3
    - ..... 
    - chinese_to_ID.json
    - ID_to_chinese.json
  - undeciphered
    - L
      - L_?_Filename
      - L_?_Filename
      - .....
    - X
      - X_?_Filename
      - .....
    - Y+H
      - Y_?_Filename
      - H_?_Filename
      - .....
  - GuoXueDaShi_1390
    - ID1
      - Source_ID1_Filename
      - Source_ID1_Filename
      - .....
    - ID2
      - Source_ID2_Filename
      - .....
    - ID3
    - ..... 
    - chinese_to_ID.json
    - ID_to_chinese.json

Source:’X’ represents "New Compilation of Oracle Bone Scripts", ’L’ represents the "Oracle Bone Script: Six Digit Numerical Code",’G’ represents the "GuoXueDaShi" website, ’Y’ represents the "YinQiWenYuan" website, and ’H’ represents the HWOBC dataset, they are the sources of the data.
## Chinese OCR
The code for training and testing (usage) is provided in the OCR folder. Includes recognition of 88,899 classes of Chinese characters. [Model download](https://figshare.com/s/7ec755b4ba77c6994ed2). Category numbers and their corresponding Chinese characters are stored in OCR/label.json. We have provided models and code with α set to 0.  
[OCR Dataset download](https://figshare.com/s/b03be2bccdd867b73e5f).  

<!-- 可以使用[train.py](OCR/train.py)进行微调或者重新训练，[Chinese_to_ID.json](OCR/Chinese_to_ID.json)和[ID_to_Chinese.json](OCR/ID_to_Chinese.json)保存OCR数据集的种类ID和汉字之间的联系，[Dataset establishment.py](<OCR/Dataset establishment.py>)用于生成训练数据集[OCR_train.json](OCR/OCR_train.json)。下载好模型后，你可以直接使用[test.py](OCR/test.py)进行测试，给了两个示例的测试图片，均是从其它pdf上裁剪下来的汉字图像。使用时候最好背景为白色，[use.json](OCR/use.json)里面是测试的图片路径，以列表形式保存，输入识别内容[result.json](OCR/result.json)。 -->

You can use [train.py](OCR/train.py) for fine-tuning or retraining. [Chinese_to_ID.json](OCR/Chinese_to_ID.json) and [ID_to_Chinese.json](OCR/ID_to_Chinese.json) store the mappings between OCR dataset category IDs and Chinese characters. [Dataset establishment.py](<OCR/Dataset establishment.py>) is used to generate the training dataset [OCR_train.json](OCR/OCR_train.json). Once the model is downloaded, you can directly use [test.py](OCR/test.py) for testing, which includes two example test images that are Chinese character images cropped from other PDFs. It's best to use images with a white background. [use.json](OCR/use.json) contains the paths to the test images, saved in a list format. The recognized content is output to [result.json](OCR/result.json).


## MoCo
The code for training and testing (usage) is provided in the MoCo folder. [Model download](https://figshare.com/s/30c206b1d1f1870ae76f).  

<!-- 可以使用[train.py](MoCo/train.py)进行微调或者重新训练，[Dataset establishment.py](<MoCo/Dataset establishment.py>)用于生成训练数据集[MOCO_train.json](MoCo/MOCO_train.json)。下载好MoCo模型后，[test.py](MoCo/test.py)用来使用MoCo，没有融合的 1,781个种类的甲骨文，寻找另一个不同种类的甲骨文的相似度大于args.w的第一个样本，用来寻找不同种类甲骨文之间的相似度，结果保存为[result.json](MoCo/result.json)。 -->

You can use [train.py](MoCo/train.py) for fine-tuning or retraining, [Dataset establishment.py](<MoCo/Dataset establishment.py>) is used to generate the training dataset [MOCO_train.json](MoCo/MOCO_train.json). After downloading the MoCo model, [test.py](MoCo/test.py) is utilized for operating MoCo on 1,781 unmerged categories of oracle bones, seeking the first sample from another category with a similarity greater than args.w to find the similarity between different categories of oracle bones. The results are saved in [result.json](MoCo/result.json). 
## Validation
The code for training and testing (usage) is provided in the Validation folder. [Model download](https://figshare.com/s/21da51dae0a5ef4b408e).  

<!-- [Dataset establishment.py](<Validation/Dataset establishment.py>)用来划分数据集，由于分类模型无法识别没见过的种类，所以把所有只有一个样本的类归入测试集。[Validation_test.json](Validation/Validation_test.json)和[Validation_train.json](Validation/Validation_train.json)分别是测试集和训练集，划分2：8。[standard deviation.py](<Validation/standard deviation.py>)用来获得训练集的标准差。
可以使用[train.py](Validation/train.py)进行微调或者重新训练。下载好模型后可以用[test.py](Validation/test.py)进行验证测试集的准确率为94.3%。[log.csv](Validation/log.csv)记录了每个epoch的训练集准确率和测试集准确率变化。 -->

[Dataset establishment.py](<Validation/Dataset establishment.py>) is used for splitting the dataset. Since the classification model cannot recognize unseen categories, all categories with only one sample are allocated to the test set. [Validation_test.json](Validation/Validation_test.json) and [Validation_train.json](Validation/Validation_train.json) are the test and training sets, respectively, split in a 2:8 ratio. [standard deviation.py](<Validation/standard deviation.py>) is used to obtain the standard deviation of the training set.

You can use [train.py](Validation/train.py) for fine-tuning or retraining. Once the model is downloaded, you can use [test.py](Validation/test.py) to validate the test set with an accuracy of 94.3%. [log.csv](Validation/log.csv) records the changes in training set accuracy and test set accuracy for each epoch. 
[Validation_label.json](Validation/Validation_label.json) stores the relationship between classification IDs and dataset category IDs.



## Instructions for use
To use MoCo or Validation, you need to download HUST-OBS. You can then directly use their trained models for prediction. If you want to use Chinese OCR, please download the OCR dataset and the corresponding model. After downloading, organize the data as follows.
 <!-- Just a reminder, after extraction in Windows, there might be a nested folder. For instance, within HUST-OBS, there could be another HUST-OBS, resulting in an additional layer of folders. Resolving this issue should enable normal usage. -->
- HUST-OBS-mian
  - [HUST-OBS](https://figshare.com/s/8a9c0420312d94fc01e3)
    - deciphered
    - ...
  - MoCo
    - [model_last.pth](https://figshare.com/s/30c206b1d1f1870ae76f)
    - ...
  - OCR
    - [OCR_Dataset](https://figshare.com/s/b03be2bccdd867b73e5f)
    - [model_last.pth](https://figshare.com/s/7ec755b4ba77c6994ed2)
    - ...
  - Validation
    - [checkpoint_ep0600.pth](https://figshare.com/s/21da51dae0a5ef4b408e)
    - ...
