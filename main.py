from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

#data_set = fetch_openml ('mnist_784', version=1)
#x,y = data_set ["data"],data_set["target"]
#print(x.shape)     #each image is 28x28 thus 784 features
#print(y.shape)
#this is to check the first element of the reshaped 784 point (or 28x28 matrix) of element 0 of the array x.
#plt.imshow (x [0].reshape(28,28),cmap = mpl.cm.binary,interpolation = "nearest")
#plt.axis("off")
#plt.show()
data_set = []
#train_set, train_set_results, test_set, test_set_results = []

#def data_aquisition():
data_set = fetch_openml('mnist_784', version=1)
x, y = data_set["data"], data_set["target"]
#seperating the dataset into training and testing
train_set, train_set_results, test_set, test_set_results = x[:60000], y[:60000], x[60000:], y[60000:]


#this is to see the stochastic gradient classifer (a binary classifier) perform on a multiclass classifier
def single_classifier_exp():
    sgc = SGDClassifier(random_state=42)
    sgc.fit(train_set, train_set_results)  #the fit method is the trainer (needs the full training set and results)
    #sgc.predict([test_set,test_set_results])
    sgc.classes_
    pred = sgc.predict(test_set)
    specific_digit = sgc.decision_function(test_set)      #decision function is the part that decides whether a choice is made
    arr = []
    print (len(specific_digit))
    with open ("results.txt","w",encoding='utf-8') as text_file:
        count = 0
        for i in range(len(test_set_results)):
            text_file.write(str(test_set_results[i])+", "+str(specific_digit[i])+", "+str(pred[i])+"\n")



single_classifier_exp()
