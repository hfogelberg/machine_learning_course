# Data Preprocessing Template

# Importing the libraries
import numpy
import matplotlib.pyplot
import pandas
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.cross_validation import train_test_split

# import the dataset
dataset = pandas.read_csv('Data.csv')
matrix = dataset.iloc[:, :-1].values
has_purchased = dataset.iloc[:, 3].values

# Take care of missing data
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(matrix[:, 1:3])
matrix[:, 1:3] = imputer.transform(matrix[:, 1:3])

# Encode categorical data
le_country = LabelEncoder()
le_country.fit_transform(matrix[:, 0])
print(le_country.classes_)
matrix[:, 0] = le_country.fit_transform(matrix[:, 0])

le_has_purchased = LabelEncoder()
has_purchased = le_has_purchased.fit_transform(has_purchased)

# Create dummy classes for countries
hotEnc = OneHotEncoder(categorical_features = [0])
matrix = hotEnc.fit_transform(matrix).toarray()

# Split dataset between Traing set and Test set
matrix_train, matrix_test, has_purchased_train, has_purchased_test = train_test_split(matrix, has_purchased, test_size = 0.2, random_state = 0)

# Feature scaling
# (Done automatically in many libabries)
sc_matrix = StandardScaler()
matrix_train = sc_matrix.fit_transform(matrix_train)
matrix_test = sc_matrix.transform(matrix_test)

