Artificial Intelligence
Sessions
	Module 1
Sessions 1	Introduction + PPT1  Python
Sessions 2	variables,data type, string + PPT1 (20th page
Sessions 3	Python Nasics (loop tuple iteration  ,list,dictionary)
Sessions 4	Library + PPT1(15:00)  + PPT 2 numpy(30:00)
Sessions 5	assignment solve + data visualisation
Sessions 6	data visaulisation + ppt (graph)+ Pandas basics + hotel  booking.csv
 	Module 2
Sessions 7	basics of sk learn Linear regression PPT
Sessions 8	Simple Linear multi logistic Regression (Salary_Data.csv)15:30 PPt(Scikit Learn)49:00  1:19 Problem!!!
Sessions 9	KNN + PPT + Underweight.Normal.xlsx + Decision Tree
Sessions 10	Decsion tree + Heart.csv+
Sessions 11	Random Forest + Naive Byes + tennis csv +svm
Sessions 12	SVM + Unsupervised Learning knn
Sessions 13	knn + data set+ keras ppt+ deep learning keras ppt   + second Project
.             Module 3
Sessions 14	neuron an-ppt  keras  tensor flow
Sessions 15	overview of keras   loss function
Sessions 16 
Sessions 17
Sessions 18
				Artificial Intelligence
(35 hrs)

		Module 1 Python (10 hrs) // 1 Minor Project

1.Basic Of Python
	- variables,data type, string
	= loop& condition
	- tuple iteration  ,list,dictionary
2. Pandas
	- data frame
3. Numpy
	-data set
4. Data Visualization
	-Mat Plot lib 
              - graph: line , scatter , bar , pie , histogram , density kernel
              - line plot uses : regression, trend analysis, forecasting





		Module 2 Machine Learning (12 hrs) 

Scikit-learn for Machine Learning 
 1. Supervised Learning
                 -Regression (C)
                               - Linear            // assignment 1
		- Multiple
             - Classification (D)
                              - Logistic 
		- KNN
		- DT decision tree/RF
		- Naive Bayes
		- SVM()
2. Unsupervised Learning           // Major Project 2
	              - Clustering
		-K means
		-hierarchal
	               - Dimensionality Reduction






Module 3 Deep Learning (12 Hrs)

1. Keras For deep learning
	 -ANN
	- CNN
	- RNN/LSTM





















			Projects 
	Numpy

Assignment 1  	Session 5(5:00) Find out Eculidean Distance between 2 points 
			related to ML algo such as KNN K-Means
Assignment 2		Perform Normalization to RBing Values in the range of 0 to 1
Assignment 3		Find the dot product of 2 matrices
Assignment 4		Find the Mean Squared Error b/w Predicted values and real values
	
	Multi Linear Regression

Assignment 1    DONE
Minor 		Session 9/10 before major
Major           Session 7 (1:17:00)(he said minor project 1st project) DONE
Project 2	Session 13 (He said its 2nd  as major project) DONE
Project 3







Explanation

		Module 1
Components
	-Function
	-Variable
Module is Python File
Library is Collection of Modules
Package is collection of Libraries

Types of Lib
	-Inbuilt: Python standard lib
	-eg : mayh.datetime,statistics,csv,json,pickle

	-User Defined lib we have tp install it 
		- Numpy (Numerical Computation with arrays)
		- Matplotlib ( Data visualization)
		- Pandas ( Data manipulation/ data processing)
			#Pandas is a Open source library for analytics
			#Data Ingestion / Data Collection
			#Data Manipulation / Data Preprocessing
			  #1. DataFrame 12 Dimensional Object in Pandas ==>Matrix Array
			  #2. Series 1 Dimensional object in Pandas ==>


		Module 2

1.Simple Linear Regression
	# for scikit learn x values should be 2D metrix
	# Plotting the best fit line

2.Multiple Linear Regression
	#multiple features
	#one output

	#best fit line for linear regression

	#y=mx+c
	#y=m1x1+m2X2+m3x3+.....mnxn+c

	#m1,m2,m3,.....mn = Slope
	# c= mutli intercepts

	# Split data into training and testing sets
	# Train the model
	# Predict for testing data
	# Evaluate the model
	# Plot the data and the best-fit line



3.Logistic Regression




4. KNN
	#Euclidean Distance
	# Test data
	#lazy alogrithm consumes more computation time for testing than train
	#KNN imputer    filling the missing parts

5. Decision Tree
	training and testing 
	metrics
	Random Forest (Ensemle Technique)
	over fit underfit best fit

Random Forest Classifier 

Ensemble Learning (Combining multiple Model)
- Bagging (Bootstrap Aggregation)
Ex: Random Forest, Bagging Classifier
- Boosting (Combines weak learners with strong learners by creating sequential models)
Ex: Gradient Boost, XGBoost, Adaboost
During the usage of DT, we tend to get overfitted model. Overfitting: Training is good, but testing is bad.

To avoid overfitting, we use Random Forest


6. Naive Byes
	#normalised probability
	#Using scikit learn lib
	#Converting the input into numerical values using encoding

7. SVM Simple Variable Model
	extension to logistic regression
	type
		seperaable
		non linear 
	Marginal plane 
 	soft vs hard margin



8. K Means
	i Similar to KNN(Distance Formula)
	# K Means is an unsupervised learning algo
	# It has unlabeled data (only ip is directly mentioned)
	# It computes the centroids and repeats until the optimal centroid is found
	# it assumes how many clusters(K) are there
	# Elbow method : Used to find the value of k

	#sum of squared distance b/t data points and centroid is min


	# step 1: Find the k value using elbow method
	# step 2 choose k data point at random  and assign it to each clusters
	# step 3 compute the centroid of the cluster
	# Step 4 add all the points one by one and start segregation into various 	clusters(distance formula)
	# Step 5 repeat step4 till optimun value of centroid
















		Module 3 Deep Learning
Basics of Deep Learning
	-ANN Artificial Neural Networks
	-CNN Convolution Neural Network
	-RNN Recurrent Neural Networks
	- LSTM Long Short Term Memory

Deep 
-supervised 
	Regression
		-simple linear regression (y=mx+c)   weight is slope and bias is intercepts
	classification
-Unsupervised 
	Auto encoders
-Reinforcement

Artificial Perceptron’
Input    - pixels of images as neurons
Weight   slope
Bias         intercept
Activation func    working 
output

feedforward  
Back propagation      adjust weights and biases to reduce error (‘loss’) in the output using optimizee
Activation Function 
ReLU	-	Rectified Linear Unit    hidden layer and regression output
Sigmoid   	graph has   shape    binary classification
Softmax	extension to sigmoid output layer for multi class classification
Linear		
Loss function
Evaluating how well algo works
-Mean squared error
