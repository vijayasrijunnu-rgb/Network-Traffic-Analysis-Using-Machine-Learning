from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

accuracy = []
X = Y = dataset = None
X_train = X_test = y_train = y_test = None

main = tkinter.Tk()
main.title("Network Traffic Analysis Using Machine Learning")
main.geometry("1300x1200")

class_labels = ['BROWSING','CHAT','FT','MAIL','P2P','STREAMING','VOIP',
                'VPN-BROWSING','VPN-CHAT','VPN-FT','VPN-MAIL',
                'VPN-P2P','VPN-STREAMING','VPN-VOIP']

def uploadDataset():
    global dataset
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="Dataset")
    text.insert(END, filename + " loaded\\n\\n")
    dataset = pd.read_csv(filename)
    text.insert(END, "Dataset before preprocessing\\n\\n")
    text.insert(END, str(dataset.head()))
    label = dataset.groupby('class1').size()
    label.plot(kind="bar")
    plt.show()

def DataPreprocessing():
    global X, Y, dataset, X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    dataset.fillna(0, inplace=True)

    le = LabelEncoder()
    dataset['class1'] = pd.Series(
        le.fit_transform(dataset['class1'].astype(str))
    )

    temp = dataset.values
    X = temp[:, 0:dataset.shape[1]-1]
    Y = temp[:, dataset.shape[1]-1]

    X = normalize(X)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]

    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2
    )

def runKNN():
    global accuracy
    accuracy.clear()
    cls = KNeighborsClassifier(n_neighbors=2)
    cls.fit(X, Y)
    predict = cls.predict(X_test)
    a = accuracy_score(y_test, predict) * 100
    accuracy.append(a)
    text.insert(END, f"KNN Accuracy : {a}\\n")

def runNB():
    cls = GaussianNB()
    cls.fit(X_train, y_train)
    predict = cls.predict(X_test)
    a = accuracy_score(y_test, predict) * 100
    accuracy.append(a)
    text.insert(END, f"Naive Bayes Accuracy : {a}\\n")

def runDT():
    cls = DecisionTreeClassifier()
    cls.fit(X_train, y_train)
    predict = cls.predict(X_test)
    a = accuracy_score(y_test, predict) * 100
    accuracy.append(a)
    text.insert(END, f"Decision Tree Accuracy : {a}\\n")

def runSVM():
    cls = svm.SVC()
    cls.fit(X_train, y_train)
    predict = cls.predict(X_test)
    a = accuracy_score(y_test, predict) * 100
    accuracy.append(a)
    text.insert(END, f"SVM Accuracy : {a}\\n")

def graph():
    bars = ('KNN','Naive Bayes','Decision Tree','SVM')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, accuracy)
    plt.xticks(y_pos, bars)
    plt.title("Accuracy Comparison Graph")
    plt.show()

def GUI():
    global text
    title = Label(main, text='Network Traffic Analysis Using Machine Learning')
    title.place(x=0, y=5)

    text = Text(main, height=30, width=110)
    text.place(x=10, y=100)

    Button(main, text="Upload Network Traffic Dataset",
           command=uploadDataset).place(x=900, y=100)
    Button(main, text="Data Preprocessing",
           command=DataPreprocessing).place(x=900, y=150)
    Button(main, text="Run KNN Algorithm",
           command=runKNN).place(x=900, y=200)
    Button(main, text="Run Naive Bayes Algorithm",
           command=runNB).place(x=900, y=250)
    Button(main, text="Run Decision Tree Algorithm",
           command=runDT).place(x=900, y=300)
    Button(main, text="Run SVM Algorithm",
           command=runSVM).place(x=900, y=350)
    Button(main, text="Comparison Graph",
           command=graph).place(x=900, y=400)

    main.mainloop()

if __name__ == "__main__":
    GUI()
