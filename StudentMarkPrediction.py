import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor

df=pd.read_csv('Student_Marks.csv')


X = df.drop(columns=['Marks'])

y = df['Marks']

model = DecisionTreeRegressor()

model.fit(X,y)

predict = model.predict([[5,5.2343],[4,3.133]])

print(predict)



