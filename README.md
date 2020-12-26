 ML-based-Network-Intrusion-Detection-using-Cyber-AWS Dataset-CSE-CIC-IDS2018-to-classify-network-attacks
In this project I used Machine Learning and Deep Learning to detect various cyber-attack types. It is a classification problem and identifies whether the network traffic behavior is normal or abnormal. The dataset includes along with 80 features extracted from the captured traffic using CICFlowMeter-V3 and above  one million rows are there. This data was extracted from UNB (Canadian Institute for Cybersecurity) to classify the network attacks for binary and multi classification.	
▪	Established the problem and solution to detect the various cyber attack types
▪	I did the data set Preprocessing steps: Handling the Missing Data, Encoding Categorical Data, Splitting the data set into test set and training set and Feature Scaling or Normalizing the Dataset.
▪	Executed the Instruction Detection System model with the Random Forest Classifier, SVM , Decision Tree, KNN and LSTM - RNN algorithms with train data (80% of Data) using TensorFlow, Keras, Pandas, NumPy and Scikit -learn packages
▪	Tested the model with 20% of test data.
▪	Analyzed the accuracy results of 3 classifiers and evaluated the performance.
I also applied the feature selection technique PCA and observe performance of the model with and with out PCA.
I created a new multiclass dataset with 13 attacks (Bot, DoS attacks-SlowHTTPTest, DoS attacks-Hulk, Brute Force -Web,Brute Force -XSS, SQL Injection, DoS attacks-GoldenEye,
DoS attacks-Slowloris, Infilteration, FTP-BruteForce, SSH Bruteforce, DDOS attack-HOIC and DDOS attack-LOIC-UDP ) from individual datasets and I picked random record of each
CSE-CIC-IDS2018 data sets (Friday-02-03-2018, Friday-16-02- 2018, Friday-23-02-2018 , Thursday-15-02-2018,Thursday-1-03-2018, Wed 14-02-2018 and Wed 21-02-2018CSE-CIC-IDS2018 datasets) amd build the multiclass models with 14 class. 
In the project, I selected hyper-parameters suchas batch size, the number of epochs, learning rate, dropout, and activation function for both binary and multi-class classification. Here I took the batch size is 50, epochs are 500, loss function categorical_crossentropy, optimizer sgd, and did experiment with changing number of neurons and number of hidden layers.
