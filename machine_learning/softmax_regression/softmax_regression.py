import numpy as np
import matplotlib.pyplot as plt
import math

class SoftmaxRegression:
    def __init__(self):
        self.__dataMat=[]
        self.__labelMat=[]
        self.__M=0;self.__N=0
        self.__K=0;   # K categories
        self.__weights=[]  

    '''
        function LoadData:Open file to obtain dataset and labelset
        inputfile: name of the input file,
        split_flag:separator of each data item in the input file
    '''
    def LoadData(self,inputfile):
        with open(inputfile,'r') as f:
            for line in f.readlines():
                lineArray=line.strip().split()
                for i in range(len(lineArray)):
                    if i==0:
                        self.__dataMat.append(1.0)
                    if i==len(lineArray)-1:
                        self.__labelMat.append(int(lineArray[i]))
                    else:
                        self.__dataMat.append(float(lineArray[i]))
        self.__M=len(self.__labelMat)
        self.__K=len(set(self.__labelMat))  #catogories: K
        self.__labelMat=np.mat(self.__labelMat).T  # labelMat: M*1
        self.__dataMat=np.mat(self.__dataMat).reshape(self.__M,-1)  #dataMat: M*N
        self.__N=np.shape(self.__dataMat)[1]
        self.__weights=np.random.random_sample((self.__K,self.__N)) #weights: K*N

    #gradient descent method to train data
    #alpha:learning rate      k:training times
    def gradientDescent(self,alpha=0.005,k=1000):
        for t in range(k): #train data k times
            sum=np.mat(np.zeros((self.__K,self.__N)))
            for i in range(self.__M): #data
                division=0
                for l in range(self.__K):
                    division+=math.exp(self.__weights[l,:]*self.__dataMat[i,:].T)
                for j in range(self.__K):
                    p=math.exp(self.__weights[j,:]*self.__dataMat[i,:].T)/division
                    if self.__labelMat[i,0]==j:
                        bl=1
                    else:
                        bl=0
                    sum[j,:]+=(bl-p)*self.__dataMat[i,:]  # 1*N
            self.__weights+=alpha*sum/self.__M          # K*N
        #print self.__weights
    
    #X is a data vector, X=[x1,x2,...xn]
    def classify(self,X):
        X_Data=[1.0]
        X_Data+=X
        Y=self.__weights*np.mat(X_Data).T #K*1
        return np.argmax(Y,axis=0)

    def plotTrainData(self):
        xcord0=[];ycord0=[]
        xcord1=[];ycord1=[]
        for i in range(self.__M):
            if self.__labelMat[i,0]==0:
                xcord0.append(self.__dataMat[i,1])
                ycord0.append(self.__dataMat[i,2])
            else:
                xcord1.append(self.__dataMat[i,1])
                ycord1.append(self.__dataMat[i,2])
        fig=plt.figure()
        ax=fig.add_subplot(111)
        ax.scatter(xcord0,ycord0,s=20,c='red',marker='x')
        ax.scatter(xcord1,ycord1,s=20,c='blue',marker='s')

        plt.title('Training Data')
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.show()
    
    def test(self):
        xcord0=[];ycord0=[]
        xcord1=[];ycord1=[]
        for i in range(100):
            x=np.random.uniform(-4.0,4.0)
            y=np.random.uniform(-5.0,15.0)
            c=self.classify([x,y])
            if c==0:
                xcord0.append(x)
                ycord0.append(y)
            else:
                xcord1.append(x)
                ycord1.append(y)
        fig=plt.figure()
        ax=fig.add_subplot(111)
        ax.scatter(xcord0,ycord0,s=20,c='red',marker='x')
        ax.scatter(xcord1,ycord1,s=20,c='blue',marker='s')
        plt.title('Test')
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.show()

if __name__=="__main__":
    r=SoftmaxRegression()
    r.LoadData("data.txt")
    r.gradientDescent()
    r.plotTrainData()
    r.test()
                        
