import numpy as np
import pandas as pd 
#đọc file dữ liệu
df = pd.read_csv("utv.csv")
X = df.drop(["diagnosis"], axis = 1)
y = df["diagnosis"]

#chia tỉ lệ train-test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

#Su dung model
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0, max_iter=500)

#train model
log_model2 =classifier.fit(X_train, y_train)
a=[[]]
a[0].append(16.65)
a[0].append(21.38)
a[0].append(110.0)
a[0].append(904.6)
a[0].append(0.1121)

#chạy mô hình học máy
print(X_train)
print(y_train)
y_pred1 = log_model2.predict(a)
print("Du doan cho du lieu vua nhap la:",y_pred1)
y_pred2 = log_model2.predict(X_test)
print("tap y du doan")
print(y_pred2)
print("<======================>")


#Đánh giá mô hình dựa trên kết quả dự đoán (với độ đo đơn giản Accuracy, Precision, Recall, F1-Score))
#In ra kết quả độ chính xác
from sklearn.metrics import accuracy_score
print("Accuracy Score: ", accuracy_score(y_test, y_pred2))

#In ra kết quả độ chính xác trên từng lớp yes/no
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred2))

#In ra ma trận kết quả dự đoán
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred2)
print("Ma tran nham lan:\n",confusion_matrix)


#Hien thi du lieu chua du doan
import matplotlib.pyplot as plt
X0 = np.array([(X_test[['mean_radius','mean_texture']].values[i]) for i in range(len(y_test)) if y_test.values[i]==0])
X1 = np.array([(X_test[['mean_radius','mean_texture']].values[i]) for i in range(len(y_test)) if y_test.values[i]==1])
plt.xlabel('mean_radius')
plt.ylabel('mean_texture')
plt.scatter(X0[:,0],X0[:,1],c='blue')
plt.scatter(X1[:,0],X1[:,1],c='red')
plt.title('Chua du doan')
plt.plot()
plt.show()


#Hien thi du lieu da du doan
x1 = np.array([(X_test[['mean_radius','mean_texture']].values[i]) for i in range(len(y_test)) if y_pred2[i]==1])
x0 = np.array([(X_test[['mean_radius','mean_texture']].values[i]) for i in range(len(y_test)) if y_pred2[i]==0])
plt.xlabel('mean_radius')
plt.ylabel('mean_texture')
plt.scatter(x0[:,0],x0[:,1],c='blue')
plt.scatter(x1[:,0],x1[:,1],c='red')
plt.title('Da du doan')
plt.plot()
plt.show()




