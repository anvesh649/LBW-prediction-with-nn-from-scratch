import pandas as pd
from sklearn.impute import SimpleImputer

df = pd.read_csv('LBW_Dataset.csv')

#replacing the missing values with mean for numerical and mode for categorical

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
df.iloc[:,[1,2,4,6]] = imputer.fit_transform(df.iloc[:,[1,2,4,6]])

imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
df.iloc[:,[3,7,8]]=imputer.fit_transform(df.iloc[:,[3,7,8]])

#normalize
scaler_object=MinMaxScaler()
df.iloc[:,[1,2,4,6]]=scaler_object.fit_transform(df.iloc[:,[1,2,4,6]])
