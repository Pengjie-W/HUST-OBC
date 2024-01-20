# HUST-OBS
Oracle Bone Script data collected by VLRLab of HUST
We have open-sourced the HUST-OBS dataset and the models used in the dataset, including: Chinese OCR, MoCo, and the ResNet50 for Validation.
## HUST-OBS Dataset
[HUST-OBS Download]
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
    - .....
  - .....

Source:’X’ represents "New Compilation of Oracle Bone Scripts", ’L’ represents the "Oracle Bone Script: Six Digit Numerical Code",’G’ represents "GuoXueDaShi" website, ’Y’ represents the "YinQiWenYuan" website, and ’H’ represents the HWOBC dataset, they are the sources of the data.
## Chinese OCR
The code for training and testing (usage) is provided in the OCR folder. Includes recognition of 88,899 classes of Chinese characters. [Model download](https://figshare.com/s/7ec755b4ba77c6994ed2). Category numbers and their corresponding Chinese characters are stored in OCR/label.json. We have provided models and code with α set to 0.
## MoCo
The code for training and testing (usage) is provided in the MoCo folder.
## Validation
The code for training and testing (usage) is provided in the Validation folder.