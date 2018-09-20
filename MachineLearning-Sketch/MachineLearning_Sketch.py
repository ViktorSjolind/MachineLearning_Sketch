'''

What is machine learning?

CLASSIFIER:
- takes data as input and assigns LABEL as output
- box of rules

Learning algorithm:
- procedure that creates classifier



SUPERVISED LEARNING:
-creates CLASSIFIER by finding patterns in examples

LABEL:

Recipe:
-collect training data
-train classifier
-make predictions

Decision tree: 
-type of classifier
-

'''

from sklearn import tree
from sklearn import datasets
iris = datasets.load_iris()
print(iris.feature_names)

features = [[140, 1], [130, 1], [150, 0], [170, 0]]
labels = [0, 0, 1, 1]
classifier = tree.DecisionTreeClassifier()
# fit = Find Patterns in Data
classifier = classifier.fit(features, labels)
print(classifier.predict([[150, 0]]))