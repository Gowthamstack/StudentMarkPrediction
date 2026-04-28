import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

df=pd.read_csv('music.csv')

X=df.drop(columns=['genre'])

y=df['genre']

model=DecisionTreeClassifier()

model.fit(X,y)

tree.export_graphviz(model,out_file="music-recommender.dot",feature_names=['age','gender'],class_names=sorted(y.unique()),label='all',rounded=True,filled=True)


predict=model.predict([[40,1],[19,1]])

print(predict)


