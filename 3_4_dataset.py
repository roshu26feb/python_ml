# Load libraries
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)

# shape
print(dataset.shape)
#You should see 150 instances and 5 attributes:

# head
print(dataset.head(20))
#You should see the first 20 rows of the data:

# descriptions
print(dataset.describe())
#We can see that all of the numerical values have the same scale (centimeters) and similar ranges between 0 and 8 centimeters

# class distribution
print(dataset.groupby('class').size())
#We can see that each class has the same number of instances (50 or 33% of the dataset).