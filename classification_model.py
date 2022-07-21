#Import libraries
import sklearn.ensemble
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import pickle
import matplotlib.pyplot as plt


penguins = pd.read_csv('penguins.csv')

# This model was developed following the procedure described by PRATIK MUKHERJEE
# https://www.kaggle.com/pratik1120/penguin-dataset-eda-classification-and-clustering


df = penguins.copy()
target = 'species'
encode_features = ['sex','island']

#get_dummies is one-hot encoding but sklearn. preprocessing. 
#LabelEncoder is incremental encoding, such as 0,1,2,3,4,... 
#one-hot encoding is more suitable for machine learning

for col in encode_features:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]

#Encode target feature
target_map = {'Adelie':0, 'Chinstrap':1, 'Gentoo':2}
def target_encode(val):
    return target_map[val]

df['species'] = df['species'].apply(target_encode)

# Separating X and y data
X = df.drop('species', axis=1)
y = df['species']

# Build RandomForestClassifier or LogisticRegression()
clf = RandomForestClassifier()

#Scaling data
from sklearn import preprocessing
X = preprocessing.scale(X)

#Splitting the data
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=13)

#fitting the data and prediction
model = clf.fit(X_train, y_train)
pred = model.predict(X_test)

# Saving the model
import pickle
pickle.dump(clf, open('penguins_clf.pkl', 'wb'))

# Checking hte performance of model
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score

print('CONFUSION MATRIX')
print(confusion_matrix(y_test, pred))

print('CLASSIFICATION REPORT\n')
print(classification_report(y_test, pred))

# Plot ROC curve

# ROC CURVE
print('ROC CURVE')
train_probs = model.predict_proba(X_train)
train_probs1 = train_probs[:, 1]
fpr0, tpr0, thresholds0 = roc_curve(y_train, train_probs1)

test_probs = model.predict_proba(X_test)
test_probs1 = test_probs[:, 1]
fpr1, tpr1, thresholds1 = roc_curve(y_test, test_probs1)

plt.plot(fpr0, tpr0, marker='.', label='train')
plt.plot(fpr1, tpr1, marker='.', label='validation')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()