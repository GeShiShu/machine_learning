import os
import math

class NaiveBayes():
    def __init__(self):
        self.__words_dict=dict()
        self.__category=[]
        self.__nums=[]
    
    def __loadData(self,filename):
        wordMat=[]
        i=1
        s_format="\'\",.;?()"
        with open(filename,'r') as f:
            for line in f.readlines():
                lineArray=line.split()
                for word in lineArray:
                    word=word.strip(s_format)
                    if word!='':
                        wordMat.append(word)
        return wordMat 

    def __getDict(self,dir):
        for category_name in os.listdir(dir): #each class name
            self.__category.append(category_name)
        for ci in range(len(self.__category)):  # class ci  (ci=0,1,..)
            length=0
            path=dir+'/'+self.__category[ci]
            files=os.listdir(path)
            self.__nums.append(len(files)) 
            for file in files:
                filepath=path+"/"+file
                wordMat=self.__loadData(filepath)
                words_set=set(wordMat)
                '''for word in wordMat:
                    if word in words_dict:
                        words_dict[word][ci]+=1
                    else:
                        lst=[0]*len(category)
                        lst[ci]=1
                        words_dict[word]=lst  '''
                for word in words_set:
                    if word in self.__words_dict:
                        self.__words_dict[word][ci]+=1
                    else:
                        lst=[0]*len(self.__category)
                        lst[ci]=1
                        self.__words_dict[word]=lst

    def test(self,test_dir):
        test_category=[]
        err=[]
        sum=[]
        index=-1
        for dir in os.listdir(test_dir):
            test_category.append(dir)
            path=test_dir+"/"+dir
            files=os.listdir(path)
            index+=1
            err.append(0.0)
            sum.append(len(files))
            for filename in files:
                filepath=path+"/"+filename 
                ans=self.predict(filepath)
                if ans!=dir:
                    err[index]+=1.0
        for i in range(len(err)):
            err[i]=err[i]/sum[i] 
            print "error of "+test_category[i]+" is ",err[i]

    def predict(self,filename):
        p=[0]*len(self.__category)
        test_wordMat=self.__loadData(filename)
        total_category=len(self.__category)  # total number of class 
        total_samples=0               # total number of samples
        for i in self.__nums:     
            total_samples+=i
        index=0;
        for k in range(len(p)):  # class k 
            npjk=1.0
            for word in test_wordMat:
                if word not in self.__words_dict: 
                    pjk=math.log10(1.0/(self.__nums[k]+total_category))  #Laplace smooth
                else:
                    pjk=math.log10(1.0*(self.__words_dict[word][k]+1.0)/(self.__nums[k]+total_category))  #Laplace smooth
                npjk=npjk+pjk
            p[k]=npjk+math.log10(1.0*self.__nums[k]/total_samples)
            if k==0:
                possibility=p[k]
            elif p[k]>possibility:
                possibility=p[k]
                index=k        # index of max possibility

        ans=self.__category[index]  # most possible class eg: pos,neg
        return ans  

    def train(self,train_dir):
        self.__getDict(train_dir)

if __name__=="__main__":
    train_dir="/home/evan/cs/courses/pattern_recognition/machine_learning/naive_bayes/tokens/train"
    test_dir="/home/evan/cs/courses/pattern_recognition/machine_learning/naive_bayes/tokens/test"
    filename=test_dir+"/neg/"+"cv631_tok-20288.txt"
    s=NaiveBayes()
    s.train(train_dir)
    #s.test(test_dir)
    ans=s.predict(filename)
    print ans
