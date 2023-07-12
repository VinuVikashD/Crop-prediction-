# import numpy as np
# import pandas as pd
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
# import warnings
# import pickle
# warnings.filterwarnings("ignore")

# data = pd.read_csv("Crop_dataset.csv")
# data = np.array(data)

# X = data[1:, 1:-1]
# y = data[1:, -1]
# y = y.astype('int')
# X = X.astype('int')
# # print(X,y)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
# log_reg = LogisticRegression()


# log_reg.fit(X_train, y_train)

# pickle.dump(log_reg,open('model.pkl','wb'))

import numpy as np
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import warnings
import pickle
warnings.filterwarnings("ignore")

dataset = pd.read_csv("Crop_dataset.csv")
shuffled_data = dataset.sample(frac=1,random_state=42).reset_index(drop=True)
shuffled_data.to_csv('shuffled_data.csv',index=False)
shuffled_data = np.array(shuffled_data)

x = shuffled_data[:, :-1]
y = shuffled_data[:, -1] 

L = LabelEncoder()
y = L.fit_transform(y)

y = y.astype('int')
x = x.astype('int')

sc = StandardScaler()
x = sc.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.2)

dt_classifier = DecisionTreeClassifier()
dt_classifier.fit(x_train, y_train)
pred_dt = dt_classifier.predict(x_test)
np.random.seed(40)
print(accuracy_score(y_test, pred_dt))


# pickle.dump(dt_classifier,open('model.pkl','wb'))
# model=pickle.load(open('model.pkl','rb'))
with open('model/model.pkl','wb') as f:
    pickle.dump(dt_classifier,f)
