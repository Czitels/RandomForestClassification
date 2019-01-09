# Classification of chocolate features
I used this dataset:
https://www.kaggle.com/rtatman/chocolate-bar-ratings
My aim was to classify chocolate by rating. I remove 4 columns: REF, Speficic Bean Origin, Review Date and Bean Type. 
I decided to classify chocolate with rating 3.7-5 (set as a 1 for RF) as a tasty chocolate and 1-3.7(set as a 0 for RF) as a bad chocolate.
RandomForest after optimize give me a 82% accurate for training and test data. 
The most interesting result is a feature importance of chocolate features.
