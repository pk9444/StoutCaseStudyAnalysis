# import libraries
import numpy as np
import pandas as pd
from scipy.spatial import distance
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_predict
import pickle

# enter the absolute path. Change it accordingly in your system
path = r'absolute path to the dataset csv file'

# read all the data into a pandas dataframe
dataframe = pd.read_csv(path)
#print(dataframe.head(10).to_string())

# Data cleaning
# Drop all the non-numeric fields to create our Feature Set
X = dataframe.drop(dataframe.columns[[0, 1, 3, 6]], axis=1)
#print(X.head(10).to_string())

# Convert the Feature Set of numeric fields into a 2D-array
X1 = X[['amount','oldbalanceOrg', 'newbalanceOrig','oldbalanceDest','newbalanceDest','isFraud']]
X2 = dataframe['isFraud'] # we will use this dataframe to compute the KL-divergence and Cosine Similarity

# Define the target class - into which the classifier will predict a transaction based on values provided to the feature set
y1 = dataframe['isFlaggedFraud']
print(X1.head(10).to_string())
print(y1.head(10).to_string())

# Define the ML classifiers - Here we use Gaussian Naive Bayes and Logistic Regression
clf_GNB = GaussianNB()
clf_LR = LogisticRegression(max_iter=1000)

# Performing K-fold cross validation for dynamic training and testing
k = 3
kf = KFold(n_splits=k, random_state=None)


# Generate all the necessary visualizations and analysis
print("\nConfusion Matrices : ")
print("----------------------------------------------------")

# Perform prediction based on k-fold snd generate confusion matrices
y_pred_LR = cross_val_predict(clf_LR, X1, y1, cv=kf)
cm_LR = confusion_matrix(y1, y_pred_LR)
print("LR_Confusion_Matrix: " + str(cm_LR))

y_pred_GNB = cross_val_predict(clf_GNB, X1, y1, cv=kf)
cm_GNB = confusion_matrix(y1, y_pred_GNB)
print("GNB_Confusion_Matrix: " + str(cm_GNB))


print("\nClassification Results : ")
print("----------------------------------------------------")

print("\nLogistic Regression : ")
classification_report_LR = classification_report(y1, y_pred_LR)
print(classification_report_LR)
print("\nGaussian NB : ")
classification_report_GNB = classification_report(y1, y_pred_GNB)
print(classification_report_GNB)

'''
get_KL_Divergence - method to get the Kullback-Liebler Divergence value 
@param  P,Q - probability distributions 
@return KL-Divergence value 
'''
def get_KL_divergence(P,Q):
    output = 0
    DELTA = 0.00001
    P = P + DELTA
    Q = Q + DELTA
    for i in range(len(P)):
        output += P[i] * np.log(P[i]/Q[i])

    return output


# Computing correlation metrics - Cosine Similarity and KL-Divergence
print("\nCorrelation Metrics : ")
print("------------------------------------------------------------------------------------------")


print("isFraud v/s IsFlaggedFraud")
print("KL-Divergence : " + str(get_KL_divergence(X2,y1)))
cosine_similarity = distance.cosine(X2,y1)
print("Cosine Similarity : " + str(cosine_similarity))

print("oldBalanceOrg v/s newBalanceOrg")
print("KL-Divergence : " + str(get_KL_divergence(dataframe['oldbalanceOrg'], dataframe['newbalanceOrig'])))
print("Cosine Similarity : " + str(distance.cosine(dataframe['oldbalanceOrg'], dataframe['newbalanceOrig'])))

print("oldBalanceDest v/s newBalanceDest")
print("KL-Divergence : " + str(get_KL_divergence(dataframe['oldbalanceDest'], dataframe['newbalanceDest'])))
print("Cosine Similarity : " + str(distance.cosine(dataframe['oldbalanceDest'], dataframe['newbalanceDest'])))


print("Count of Transaction Types :")
print("------------------------------------------------------------------------------------------")
type_frequencies = dataframe['type'].value_counts()
print(type_frequencies)



# ML-Classifier objects to be used in Flask
# Uncomment these lines below to generate the .pkl files for building the pipeline
# clf_GNB.fit(X1, y1)
# pickle.dump(clf_GNB, open('model_GNB.pkl','wb'))
# clf_LR.fit(X1, y1)
# pickle.dump(clf_LR, open('model_LR.pkl','wb'))