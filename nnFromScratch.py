#%%
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

# %% seperate independent and dependent variables
df= pd.read_csv('heart.csv')
X = np.array(df.loc[:, df.columns !='output'])
y = np.array(df['output'])

#%% Train / Test split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=123)
#%% scale the data
scaler = StandardScaler()
x_train_scale = scaler.fit_transform(X_train)
x_test_scale = scaler.transform(X_test)

x_train_scale.shape

#%%
class NN():
    def __init__(self,LR, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.w = np.random.randn(X_train.shape[1])
        self.b = np.random.randn()
        self.LR =LR 
        self.L_train = []
        self.L_test = []

        #activation Function
    def activation(self, x):
        return 1/ (1+np.exp(-x))
        #derivative of activation function
    def dactivation(self, x):
        return self.activation(x) * (1 - self.activation(x))

        
    def forward(self,X):
        hidden_1 = np.dot(X , self.w) + self.b
        activate_1 = self.activation(hidden_1)

        return activate_1
        #Backpropagation
    def backward(self, X, y_true):
        hidden_1 = np.dot(X , self.w) + self.b
        y_pred = self.forward(X)


        
        
        dl_dpred = 2 * (y_pred - y_true)
        dpred_dhidden1 = self.dactivation(hidden_1)
        dhidden1_dw = X
        dhidden1_db = 1

        dl_dw = dl_dpred * dpred_dhidden1  * dhidden1_dw
        dl_db = dl_dpred * dpred_dhidden1  * dhidden1_db 

        return dl_db, dl_dw

    def optimizer(self, dl_db, dl_dw ):

        self.b = self.b - (self.LR * dl_db)
        self.w = self.w - (self.LR * dl_dw)

    def train(self, iteration):


        # implementation of SGD(stochastic Gradien Descent)
        for i in range(iteration):
            random_pos = np.random.randint(len(self.X_train))

            y_true = self.y_train[random_pos]   
            y_pred = self.forward(self.X_train[random_pos]) 

            L = np.sum(np.square(y_pred - y_true))
            self.L_train.append(L)

            dl_db, dl_dw = self.backward(self.X_train[random_pos], self.y_train[random_pos])

            self.optimizer(dl_db=dl_db, dl_dw=dl_dw)

            #calculate the test error for each epoc
            L_sum = 0

            for j in range(len(self.X_test)):
                y_true = self.y_test[j]
                y_pred = self.forward(self.X_test[j])

                L_sum += np.square(y_pred - y_true)
            self.L_test.append(L_sum)
        return 'Training successfully finished'    







# %% Hyper parameters
LR = 0.1
ITERATIONS = 1000

nn = NN(LR=LR,X_train=x_train_scale,y_train=y_train, X_test=x_test_scale, y_test=y_test)
nn.train(iteration=ITERATIONS)

# %%
# %%
sns.lineplot(x=list(range(len(nn.L_test))), y=nn.L_test)
# %% itterate over test data

total = x_test_scale.shape[0]
correct = 0
y_preds = []
for i in range(total):
    y_true = y_test[i]
    y_pred = np.round(nn.forward(x_test_scale[i]))
    y_preds.append(y_pred)
    correct += 1 if y_pred == y_true else 0
acc = (correct / total) * 100    

# %% confussion Matrix
confusion_matrix(y_true=y_test, y_pred=y_preds)

# %%
