# Fundus_View_Classifier

A neural network that differentiates between Fundus Eye images. 

"Bad" Eye Image 


<img width="276" alt="TN1" src="https://github.com/Tkuo42/Fundus_View_Classifier/assets/71362962/11370b29-7f0b-4938-ba6f-9ee503456a86">

"Good" Eye Image


<img width="290" alt="TP1" src="https://github.com/Tkuo42/Fundus_View_Classifier/assets/71362962/f2d7c089-fe9a-42bb-91d1-1ca7478affc8">

# How to use 

Requirements 
1. Have a csv file with a column "key" that has the names of the images, place this file in data/csv
2. Place images in folder 'data/samples'
3. Edit the csv file name in 'test.py'

To use: 
1. Open 'test.py'
2. Edit csv file name if haven't already
3. Run test.py
4. Results are stored in the "results" folder: all_labeled.csv contains all the images and their labels (whether valid or invalid), valid_images.csv contains all of the valid images and their names. 


# Results of Initial Testing: 


0.99 AUROC 


0.1 BCELoss 


Confusion Matrix: 




![confusion_matrix](https://github.com/Tkuo42/Fundus_View_Classifier/assets/71362962/4aeee856-9aa1-40e8-b7d8-f1c38366c878)
