
import pandas as pd
import numpy as np
import pickle
from sklearn.svm import SVC
#importing data
data= pd.read_csv("heart.csv")

# assigning value to x and y
X=data.drop(columns='target', axis=1)
Y=data['target']

#making model
model= SVC()

model.fit(X,Y)
print(model.predict([[50,0,0,110,254,0,0,159,0,0.0,2,0,2]]))

pickle.dump(model, open('model.pkl','wb'))
pmodel = pickle.load(open('model.pkl','rb'))