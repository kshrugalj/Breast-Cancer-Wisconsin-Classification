from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt

data = load_breast_cancer()

df = pd.DataFrame(data.data, columns=data.feature_names)
print(df.sample(5)) #shows first 5 samples of the dataset

print(df.info)#shows what type of data is takes 
print(df.describe())#show stats of the dataset (e.g. count, mean, std, etc)

df2 = pd.DataFrame(data.target, columns=['target'])
print(df2.sample(5))

#This shows the pie chart of the distribution of the dataset, 
# classCounts = df2["target"].value_counts()
# plt.pie(classCounts, labels=classCounts.index, autopct="%1.2f%%", colors = ['red', 'green'])
# plt.show()

#Splitting the dataset into training and testing data
xTrain, xTest, yTrain, yTest = train_test_split(data.data, data.target, test_size=0.33, random_state=40)

#Building and Training the Model 
model = GaussianNB()
model.fit(xTrain,yTrain)

#Making predictions
yPred = model.predict(xTest)
print(yPred[:10])

#Evaluating Model Accuracy
accuracy = accuracy_score(yTest,yPred)
print(f"Model Accuracy {accuracy*100:.2f}%")