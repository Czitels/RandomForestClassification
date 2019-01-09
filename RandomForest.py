from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

chocolate = pd.read_csv("C:\choco.csv", delimiter=',')
data_random = chocolate.sample(frac=1)

keep_col = ['Company', 'Cocoa Percent', 'Company Location', 'Broad Bean Origin']
rating_col = ['Rating']

data_keep = data_random[keep_col]
ratings = data_random[rating_col]

for index, row in ratings.iterrows():
    if(row['Rating']>3.7):
        row['Rating']=1.0
    else:
        row['Rating']=0.0

encoder = preprocessing.LabelEncoder()
ratings_encoded = encoder.fit_transform(ratings)
data_dummies = pd.get_dummies(data_keep)

X_train, X_test, y_train, y_test = train_test_split(data_dummies, ratings_encoded)

forest = RandomForestClassifier(n_estimators=500, n_jobs=-1, max_features=50, min_samples_leaf=50)
forest.fit(X_train, np.ravel(y_train))

print("Classification forest:")
print(forest.score(X_train, y_train))
print(forest.score(X_test, y_test))

featuresArray = []
sum1features = [1, 414]
sum2features = [415,475]
sum3features = [476,575]
featuresArray.append(forest.feature_importances_[0])
featuresArray.append(sum(forest.feature_importances_[sum1features]))
featuresArray.append(sum(forest.feature_importances_[sum2features]))
featuresArray.append(sum(forest.feature_importances_[sum3features]))
npArray = np.asarray(featuresArray)

labels = ['Cocoa Percent','Company', 'Company Location', 'Broad Bean Origin']
n_features = npArray.shape[0]
plt.barh(range(n_features),npArray,align='center')
plt.yticks(np.arange(n_features),labels)
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.show()