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
clf_tree = tree.DecisionTreeClassifier()
clf_tree = clf_tree.fit(X, y)

#tree_prediction = clf.predict([[190, 70, 43]])

tree_prediction = clf_tree.predict(X_new)
print("1. Prediction using Decision Tree Classifier model: ", tree_prediction)

# 2. Using Logistic Regression Classifier
clf_logistic_regression = LogisticRegression(C = 1.0, solver = 'lbfgs')
clf_logistic_regression.fit(X, y)

logistic_regression_prediction = clf_logistic_regression.predict(X_new)
print("2. Prediction using Logistic Regression model: ", logistic_regression_prediction)

# 3. Using Artificial Neural Network MLPCLassifier
clf_ann_mlp_classifier = MLPClassifier(hidden_layer_sizes = (10, 10), activation = 'relu')
clf_ann_mlp_classifier.fit(X, y)

neural_network_prediction = clf_ann_mlp_classifier.predict(X_new)
print("3. Prediction using ANN model: ", neural_network_prediction)

# 4. Using SVC (Support Vector Machine)
clf_svm = svm.SVC(kernel = 'linear', C = 1.0)
clf_svm.fit(X, y)

svc_prediction = clf_svm.predict(X_new)
print("4. Prediction using SVM: ", svc_prediction)

# 5. Using Naive Bayes (NB) - GaussianNB
clf_gaussian_nb = GaussianNB()
clf_gaussian_nb.fit(X, y)

gaussian_naive_bayes_prediction = clf_gaussian_nb.predict(X_new)
print("5. Prediction using Gaussian Naive Bayes model: ",gaussian_naive_bayes_prediction)

# 6. Using KNN (K-Nearest Neighbors)
clf_knn = KNeighborsClassifier(n_neighbors = 5)
clf_knn.fit(X, y)

knn_prediction = clf_knn.predict(X_new)
print("6. Prediction using KNN (with K = 5) model: ", knn_prediction)

# 7. Using RandomForestClassifier
clf_random_forest_classifier = RandomForestClassifier()
clf_random_forest_classifier.fit(X, y)

random_forest_classifier_prediction = clf_random_forest_classifier.predict(X_new)
print("7. Prediction using Random Forest Classifier model: ", random_forest_classifier_prediction)

# 8. Using ExtraTreesClassifier
clf_extra_trees_classifier = ExtraTreesClassifier()
clf_extra_trees_classifier.fit(X, y)

extra_trees_classifier_prediction = clf_extra_trees_classifier.predict(X_new)
print("8. Prediction using Extra Trees Classifier model: ", extra_trees_classifier_prediction)

# 9. Using Gradient Boosting Trees (GBT)
clf_gradient_boosting_trees = GradientBoostingClassifier()
clf_gradient_boosting_trees.fit(X, y)

gradient_boosting_classifier_prediction = clf_gradient_boosting_trees.predict(X_new)
print("9. Prediction using Gradient Boosting Trees (GBT): ", gradient_boosting_classifier_prediction)