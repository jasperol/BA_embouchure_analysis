# Image-Based Technique Analysis: Using Deep Learning Methods to Analyse Trumpet Embouchure Technique

This Github repo contains the code used for my bachelor thesis in Cognitive Science at the University of Osnabrück. 
The thesis was overlooked by and a workspace was provided by the Musicology department of the University of Osnabrück. 
The aim of the thesis was to develop a image based learning aid for trumpet players. Allowing objective and precise feedback 
to the user.

In order for the code to run, take the following steps:
1. create a conda environment with python version 3.7.12
2. install packages as shown in the requirements.txt file


  - using the pre_augmentation.py file you can use the preprocessing pipeline to prepare data for training or prediction
   (the data used for training is not uploaded due to data protection of the participants)
  - using the training.ipynb file you can train the pretrained models
  - in the Results.ipynb file you can use the pretrained models and load the task specific
    weights, that can be found in the weights folder
      - The naming convention of the weights is: weights_ **Model** _ **Task** _ **Batchsize** _ **Epochs** 

