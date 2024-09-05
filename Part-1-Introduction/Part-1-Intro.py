from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier

#[height, weight, shoe size]
# 11 samples, 3 features

X = [[181, 80, 40], 
	 [177, 70, 43], 
	 [160, 60, 38], 
	 [154, 54, 37], 
	 [166, 65, 40], 
	 [190, 90, 47], 
	 [175, 64, 39], 
	 [177, 70, 40], 
	 [159, 55, 37], 
	 [171, 75, 42], 
	 [181, 85, 43]]

y = ['Male', 'Female', 'Female', 'Female', 'Male', 'Male', 'Male', 'Female', 'Male', 'Female', 'Male']

# X_new - Data to be used for prediction
X_new = [[190, 70, 43]]

# 1. Using Decision Tree Classifier
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, y)

#tree_prediction = clf.predict([[190, 70, 43]])

tree_prediction = clf.predict(X_new)
print("1. Prediction using Decision Tree Classifier model: ", tree_prediction)

# 2. Using Logistic Regression Classifier
clf_2 = LogisticRegression(C = 1.0, solver = 'lbfgs')
clf_2.fit(X, y)

logistic_regression_prediction = clf_2.predict(X_new)
print("2. Prediction using Logistic Regression model: ", logistic_regression_prediction)

# 3. Using Artificial Neural Network MLPCLassifier
clf_3 = MLPClassifier(hidden_layer_sizes = (10, 10), activation = 'relu')
clf_3.fit(X, y)

neural_network_prediction = clf_3.predict(X_new)
print("3. Prediction using ANN model: ", neural_network_prediction)

# 4. Using SVC (Support Vector Machine)
clf_4 = svm.SVC(kernel = 'linear', C = 1.0)
clf_4.fit(X, y)

svc_prediction = clf_4.predict(X_new)
print("4. Prediction using SVM: ", svc_prediction)

# 5. Using Naive Bayes (NB) - GaussianNB
clf_5 = GaussianNB()
clf_5.fit(X, y)

gaussian_naive_bayes_prediction = clf_5.predict(X_new)
print("5. Prediction using Gaussian Naive Bayes model: ",gaussian_naive_bayes_prediction)

# 6. Using KNN (K-Nearest Neighbors)
clf_6 = KNeighborsClassifier(n_neighbors = 5)
clf_6.fit(X, y)

knn_prediction = clf_6.predict(X_new)
print("6. Prediction using KNN (with K = 5) model: ", knn_prediction)

# 7. Using RandomForestClassifier
clf_7 = RandomForestClassifier()
clf_7.fit(X, y)

random_forest_classifier_prediction = clf_7.predict(X_new)
print("7. Prediction using Random Forest Classifier model: ", random_forest_classifier_prediction)

# 8. Using ExtraTreesClassifier
clf_8 = ExtraTreesClassifier()
clf_8.fit(X, y)

extra_trees_classifier_prediction = clf_8.predict(X_new)
print("8. Prediction using Extra Trees Classifier model: ", extra_trees_classifier_prediction)

# 9. Using Gradient Boosting Trees (GBT)
clf_9 = GradientBoostingClassifier()
clf_9.fit(X, y)

gradient_boosting_classifier_prediction = clf_9.predict(X_new)
print("9. Prediction using Gradient Boosting Trees (GBT): ", gradient_boosting_classifier_prediction)

