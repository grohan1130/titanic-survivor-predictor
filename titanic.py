import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


train_data = pd.read_csv('titanic-survivor-predictor/data/input/train.csv')
#print(train_data.head())

test_data = pd.read_csv('titanic-survivor-predictor/data/input/test.csv')
#print(test_data.head())

women = train_data.loc[train_data.Sex == 'female']['Survived']
rate_women = sum(women)/len(women)
print(rate_women)

men = train_data.loc[train_data.Sex == 'male']['Survived']
rate_men = sum(men)/len(men)
print(rate_men)


y = train_data["Survived"]

features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('titanic-survivor-predictor/data/output/submission.csv', index=False)