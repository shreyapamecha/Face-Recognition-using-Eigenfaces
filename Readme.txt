### Face Recognition using Eigenfaces #### 

Algorithm Used: Developed by Vision and Modelling(https://ieeexplore.ieee.org/document/139758) group    
Database Used: AT & T dataset(https://www.kaggle.com/kasikrit/att-database-of-faces/version/1#_=_)

The dataset is divided into 2 parts: 'training'(70 %)(280 images) and 'testing'(30 %)(120 images). 
It contains 10 pictures of 40 different people. 

Before running the code, all the paths should be specified to 'training_path' and 'testing_path' variables. 
On running the code (Code.py), two figures will appear displaying mean face image and the eigenfaces for a specified variable 'k'(20).
You can change the value of 'k' which takes the most important 'k' eigenfaces into consideration.
Now, the images from the 'test' folder are taken and their respective classes are displayed. 
At last, in the command prompt, the accuracy and the processing time will be displayed.

As you vary the value of 'k', your accuracy and running time will vary.

#### Thank you #####