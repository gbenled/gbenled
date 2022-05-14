import numpy as np
import requests
import time
from bs4 import BeautifulSoup
import csv
import urllib.request
import codecs
from collections import Counter
import pandas as pd
import itertools
from datetime import date, timedelta,datetime
import datetime
from sklearn.model_selection import train_test_split
import array
import gspread
from gspread_dataframe import set_with_dataframe
######################################################

def get_data():
    """
    returns powerball updated list from the internet as a pandas dataframe 
    """
    try:
        link = 'https://www.txlottery.org/export/sites/lottery/Games/Powerball/Winning_Numbers/download.html'
        r = requests.get(link)
        soup = BeautifulSoup(r.text, "html.parser")
        l= []
        for i in soup.find_all('a'):
            if i.get('href')!=None:
                if i.get('href')[0]!= "/" and i.get('href')[-3:]== "csv":
                    link = (i.get('href'))

        url = link
        response = urllib.request.urlopen(url)
        cr = csv.reader(codecs.iterdecode(response, 'utf-8'))
        a=[]
        for row in cr:
            a.append(row)

        data = pd.DataFrame(a)
        data.to_csv('get_data.csv')
    except:
        ############################################
        def err():
            err_msg= "Database down using back up"
            return err_msg
        ############################################   
        err()
        data = pd.read_csv("get_data.csv",index_col=[0])
        data = data.rename(columns={"0":0,"1":1,"2":2, "3":3, "$":4, "5":5, "6":6, "7":7, "8":8, "9":9,"10":10})
    return data

def powerball_list(d = get_data(),kind = "s", ball = "m"):
    """
    takes in kind  = "s" . if list is to be sorted set kind = "s" 
    else set kind = "u"
    ball = "r" for the red ball
    ball = "m" for the main ball
    returns list of all powerball numbers 
    outputs result in a list of list formart
    """
    data = d
    if ball == "m":
        try:
            data.drop([0,10], axis = 1, inplace=True)
        except:
            pass
        
        data = data.astype('int32')
        data1 = data.copy()
        data1.drop([1,2,3,9], axis = 1, inplace=True)
        data1 = data1.reset_index()
        data1.drop(['index'], axis = 1, inplace=True)
        data1 = data1.values.tolist()
        new_pb_l = []
        if kind == "s":
            for lists in data1:
                new_pb_l.append(sorted(lists))
        else:
            for lists in data1:
                new_pb_l.append(lists)
        return new_pb_l
    else:
        red_ball_list = data[9].tolist()
        red_ball_list = [int(i) for i in red_ball_list]
        return red_ball_list

def p_category (arr, ans,kind):
    """
    takes in a list of powerball number as arr = list, 
    takes in ans that is a dictionary of category of hot or cold numbers ans = dict
    takes in kind as string if kind = a it means we want to categorize normally(1-17,18-34,35-71,52-69)
     if kind = f it means we want to categorize hot or cold

    """
    
    pos=0
    if kind == "a":
        while pos < len(arr):
            if arr[pos]<18:
                arr[pos]= 0
            elif arr[pos]<35 and arr[pos]>17:
                arr[pos]= 1
            elif arr[pos]<53 and arr[pos]>34:
                arr[pos]= 2
            elif arr[pos]<70 and arr[pos]>52:
                arr[pos]= 3

            pos+=1
    elif kind == "f":
        while pos < len(arr):
            if ans[arr[pos]] < 51:#hot
                arr[pos] = 0
            elif ans[arr[pos]] > 50 and ans[arr[pos]] <101: #warm
                arr[pos] = 1
            elif ans[arr[pos]] > 100: #cold
                arr[pos] = 2
            pos+=1
    return arr
def powerball_category(kind = "a"):
    """
    takes in kind as string if kind = a it means we want to categorize normally
     if kind = f it means we want to categorize hot or cold
    """
    ans = new_cate()
    new_pb_l = powerball_list()
    chec = []
    k = kind
    for i in new_pb_l:
        chec.append(sorted(p_category(i,ans, kind = k)))
    return chec

def cleaning(d= get_data(), kind = "c"): 
    """
    takes in kind as string if kind = c it means we want to clean and categorize normally
     if kind = h it means we want to clean and categorize using hot or cold
     kind = n it means we want to clean without categorizing
     return a dataframe of the cleaned powerball result
    """
    data = d
    try:
        data.drop([0,10], axis = 1, inplace=True)
    except:
        pass
    
    data = data.astype('int32')
    new_pb_l = powerball_list(data)
    n1 = []
    n2 = []
    n3 = []
    n4 = []
    n5 = []
    
    for i in new_pb_l:
        n1.append(i[0])
        n2.append(i[1])
        n3.append(i[2])
        n4.append(i[3])
        n5.append(i[4])
    
    dic = {"1st_num":n1, "2nd_num" :n2,"3rd_num": n3,"4th_num": n4,"5th_num":n5} 
    
    data1 = pd.DataFrame(dic, index=None)
    
    data.columns= ["Month", "Day", "Year", "1st num", "2nd num", "3rd num", "4th num", "5th num","Powerball"]
    data['combined']=data['Year'].astype(str)+'-'+data['Month'].astype(str)+'-'+data['Day'].astype(str)
    
    data = pd.concat([data,data1],sort=False,axis=1)
    
    data["combined" ] = pd.to_datetime(data["combined"])
    data["day of week"] = data["combined"].dt.dayofweek
    data.drop(["1st num", "2nd num", "3rd num", "4th num", "5th num"], axis = 1, inplace=True)
    def category (arr, kind, d = get_data()):
        dic = {}
        ans = new_cate(d)
        if kind =="c":
            for j in arr:
                if j<18:
                    dic[j]= 0
                if j<35 and j>17:
                    dic[j]= 1
                if j<53 and j>34:
                    dic[j]= 2
                if j<70 and j>52:
                    dic[j]= 3
        elif kind =="h":
            for j in arr:
                if ans[j] < 51:#cold
                    dic[j] = 0
                elif ans[j] > 50 and ans[j] <101: #warm
                    dic[j] = 1
                elif ans[j] > 100: #hot
                    dic[j] = 2

        return dic
    
    def year_change (arr):
        dic = {}

        for j in arr:
            if j == 2010:
                dic[j]= 0
            if j == 2011:
                dic[j]= 1
            if j == 2012:
                dic[j]= 2
            if j == 2013:
                dic[j]= 3
            if j == 2014:
                dic[j]= 4
            if j == 2015:
                dic[j]= 5
            if j == 2016:
                dic[j]= 6
            if j == 2017:
                dic[j]= 7
            if j == 2018:
                dic[j]= 8
            if j == 2018:
                dic[j]= 8
            if j == 2019:
                dic[j]= 9
            if j == 2020:
                dic[j]= 10
            if j == 2021:
                dic[j]= 11
        return dic

    
    yr = year_change(data["Year"])
    data["Year"] = data["Year"].map(yr)
    if kind == "c":
        numb1 = category(data["1st_num"], kind,d)
        numb2 = category(data["2nd_num"], kind,d)
        numb3 = category(data["3rd_num"], kind,d)
        numb4 = category(data["4th_num"], kind,d)
        numb5 = category(data["5th_num"], kind,d)
        data["1st_num"] = data["1st_num"].map(numb1)
        data["2nd_num"] = data["2nd_num"].map(numb2)
        data["3rd_num"] = data["3rd_num"].map(numb3)
        data["4th_num"] = data["4th_num"].map(numb4)
        data["5th_num"] = data["5th_num"].map(numb5)
    elif kind== "h":
        numb1 = category(data["1st_num"], kind,d)
        numb2 = category(data["2nd_num"], kind,d)
        numb3 = category(data["3rd_num"], kind,d)
        numb4 = category(data["4th_num"], kind,d)
        numb5 = category(data["5th_num"], kind,d)
        data["1st_num"] = data["1st_num"].map(numb1)
        data["2nd_num"] = data["2nd_num"].map(numb2)
        data["3rd_num"] = data["3rd_num"].map(numb3)
        data["4th_num"] = data["4th_num"].map(numb4)
        data["5th_num"] = data["5th_num"].map(numb5)
        
    return data
def year_test(j):
    
        if j == 2010:
            return 0
        if j == 2011:
            return 1
        if j == 2012:
            return 2
        if j == 2013:
            return 3
        if j == 2014:
            return 4
        if j == 2015:
            return 5
        if j == 2016:
            return 6
        if j == 2017:
            return 7
        if j == 2018:
            return 8
        if j == 2018:
            return 8
        if j == 2019:
            return 9
        if j == 2020:
            return 10
        if j == 2021:
            return 11
        
def log_reg(X,y,year, month, day,k):
    
    from sklearn.linear_model import LogisticRegression
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)
    reg = 0.1
    dow = datetime.date(year, month, day).weekday()    

    # train a logistic regression model on the training set
    multi_model = LogisticRegression(C=1/reg, solver='lbfgs', multi_class='auto', max_iter=10000).fit(X_train, y_train)            
    predictions = multi_model.predict(X_test)
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    year = year_test(year)
    acc = accuracy_score(y_test, predictions)
    dic = {"Month":[month], "Day" :[day],"Year": [year],"day of week": [dow]} 

    test = pd.DataFrame(dic, index=None)
    predictions = multi_model.predict(test)
    return (acc, predictions)
def KNN(X,y,year, month, day,k):
    from sklearn.neighbors import KNeighborsClassifier
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)
    reg = 0.1
    dow = datetime.date(year, month, day).weekday()    
    # train a logistic regression model on the training set
    knn = KNeighborsClassifier(n_neighbors=k)

    #Train the model using the training sets
    knn.fit(X_train, y_train)            
    predictions = knn.predict(X_test)
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    year = year_test(year)
    acc = accuracy_score(y_test, predictions)
    dic = {"Month":[month], "Day" :[day],"Year": [year],"day of week": [dow]} 

    test = pd.DataFrame(dic, index=None)
    predictions = knn.predict(test)
    return (acc, predictions)

def D_tree(X,y,year, month, day,k):
    
    from sklearn.tree import DecisionTreeClassifier 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)
    reg = 0.1
    dow = datetime.date(year, month, day).weekday()
    # train a logistic regression model on the training set
    dtree_model = DecisionTreeClassifier(max_depth = k).fit(X_train, y_train)             
    predictions = dtree_model.predict(X_test)
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    year = year_test(year)
    acc = accuracy_score(y_test, predictions)
    dic = {"Month":[month], "Day" :[day],"Year": [year],"day of week": [dow]} 

    test = pd.DataFrame(dic, index=None)
    predictions = dtree_model.predict(test)
    return (acc, predictions)

def models (d,year, month, day, model,k,mode):
    log_dic = {}
    for i in range(1,6):
        data = cleaning(d,kind=mode)
        
        if i == 1:
            data.drop(["5th_num", "2nd_num", "3rd_num", "4th_num", "Powerball", "combined"], axis = 1 , inplace =True)
            X = data.drop('1st_num',axis=1).values
            y = data['1st_num'].values
            ans = model(X,y,year,month,day,k)
            log_dic[str(round(ans[0]*100,1))+"%"]= int(ans[1])
            
        elif i == 2:
            data.drop(["5th_num", "1st_num", "3rd_num", "4th_num", "Powerball", "combined"], axis = 1 , inplace =True)
            X = data.drop('2nd_num',axis=1).values
            y = data['2nd_num'].values
            ans = model(X,y,year,month,day,k)
            log_dic[str(round(ans[0]*100,1))+"%"]= int(ans[1])
            
        elif i == 3:
            data.drop(["5th_num", "1st_num", "2nd_num", "4th_num", "Powerball", "combined"], axis = 1 , inplace =True)
            X = data.drop('3rd_num',axis=1).values
            y = data['3rd_num'].values
            ans = model(X,y,year,month,day,k)
            log_dic[str(round(ans[0]*100,1))+"%"]= int(ans[1])
            
        
        elif i == 4:
            data.drop(["5th_num", "1st_num", "2nd_num", "3rd_num", "Powerball", "combined"], axis = 1 , inplace =True)
            X = data.drop('4th_num',axis=1).values
            y = data['4th_num'].values
            ans = model(X,y,year,month,day,k)
            log_dic[str(round(ans[0]*100,1))+"%"]= int(ans[1])
            
            
        elif i == 5:
            data.drop(["3rd_num", "1st_num", "2nd_num", "4th_num", "Powerball", "combined"], axis = 1 , inplace =True)
            X = data.drop('5th_num',axis=1).values
            y = data['5th_num'].values
            ans = model(X,y,year,month,day,k)
            log_dic[str(round(ans[0]*100,1))+"%"]= int(ans[1])
            
    return log_dic
def Prediction(year, month, day, kind= "l",mode ="c",data = get_data()):
    """
    data1= dataset, year = year we want to predict, month=month of prediction, day = day of prediction
    kind = type of analysis, if l  =  logistic regression, n = k-nearest neighbor,d = decision tree
    \mode = is for cleaning, c =category, h= hot & cold category, n = for none
    """
    d = data
    if kind == "l":
        model = log_reg
        k = 0
    elif kind == "n":
        while True:
#             k = input("Enter number of K(default = 3): ")
            k = 5
            try:
                k = int(k)
                if k <= 0 :
                    k = 1/0
                break
            except:
                print("Invalid k, re-enter k")
        model = KNN
    elif kind == "d":
        while True:
#             k = input("Enter depth of the Tree (default = 3): ")
            k = 5
            try:
                k = int(k)
                if k <= 0 :
                    k = 1/0
                break
            except:
                print("Invalid k, re-enter k")
        model = D_tree
    return models(d, year, month, day, model,k, mode)        
    
def numbers_skipped (new_pb_l):
    """
    takes in powerball list and returns a dictionary of how many numbers was skipped and
    how many occurence a number has
    """
    pd = {}

    for i in range (1,70):
        pd[i]=[]

    tot = 0
    count = 0
    checker = 0
    na = 0

    for i in range (1,70):
        pos = 0
        tot= 0
        while pos < len(new_pb_l):
            if i in new_pb_l[pos]:
                checker+=1
                pos+=1
                count="Not Done"
            else:
                na+=1
                count = "Done"
                pos+=1

            if count == "Done" and checker > 0:
                pd[i].append(checker)
                tot+=checker
                checker = 0
            if count == "Not Done" and na > 0:
                pd[i].append("skipped ==" +str(na))
                tot+=na
                na = 0
            if pos == len(new_pb_l):
                if checker!= 0:
                    pd[i].append(checker)
                    tot+=checker
                    checker=0
                elif na!=0:
                    pd[i].append("skipped ==" +str(na))
                    tot+=na
                    na=0
            count=0
    return pd
def sum_class(new = powerball_list()):
    """
    sum of all numbers in a powerball sequnce of number
    returns a list"""
    sum_nval = []

    for i in new:
        sum_nval.append(sum(i))
    return sum_nval

def sum_total():
    """
    returns sum of the sum of all the powerbal number 
    """
    s = sum_class()
    s_new = []
    for i in s:
        t = list(str(i))
        t = [int(x) for x in  t]
        s_new.append(sum(t))
    return s_new

def stats ():
    """
    get the frequency of each number and you could plot a graph
    """

def wining_index ():
    """
    returns all index of all numbers that has won in the past
    """
    new1 = powerball_combination()
    new_pb_l = powerball_list(kind="s")
    win_index = []
    count = 0
    pool = {}
    for n in new1:
        pool["".join([str(i) for i in n])] = count
        count+=1
    for i in new_pb_l:
        win_index.append(pool["".join([str(j) for j in i])])
    return win_index

def difference_pattern():
    diff_sum = []
    pos = 0
    sum_nval = sum_class()
    while pos< len(sum_nval)-1:
        diff_sum.append(sum_nval[pos]-sum_nval[pos+1])
        pos+=1
    neg = []

    for i in diff_sum:
        if i < 0 :
            neg.append("neg")
        elif i > 0:
            neg.append("pos")
        else:
            neg.append("Neut")
    return diff_sum,neg
def length_pattern():
    lenght =[]
    red = wining_index()
    for i in red:
        lenght.append(len(str(i)))
        
    nk = [i%2 for i in red]
    return lenght, nk

def odd_even(lists):
    d = {"even":0,"odd":0}
    
    for i in lists:
        if i%2 == 0:
            d["even"]+=1
        else:
            d["odd"]+=1
            
    return list(d.values())
def upper_lower(lists):
    d = {"upper":0,"lower":0}
    
    for i in lists:
        if i <= 35:
            d["upper"]+=1
        else:
            d["lower"]+=1
            
    return list(d.values())

def even_odd_upper_l ():  
    new_pb_l = powerball_list()
    upper_l =[]
    even_o =[]
    for i in new_pb_l:
        upper_l.append(upper_lower(i))
    
    for j in new_pb_l:
        even_o.append(odd_even(j))
    return  even_o,upper_l,

def numbers_skipped_two(new_pb_l):
    """
    takes in powerball list and returns a dictionary of how many numbers was skipped and
    a list of all occurence a number has
    """
    pd2 = {}

    for i in range (1,70):
        pd2[i]=[]
        pd2[str(i)+" skipped"] = []
    tot = 0
    count = 0
    checker = 0
    na = 0

    for i in range (1,70):
        pos = 0
        tot= 0
        while pos < len(new_pb_l):
            if i in new_pb_l[pos]:
                checker+=1
                pos+=1
                count="Not Done"
            else:
                na+=1
                count = "Done"
                pos+=1

            if count == "Done" and checker > 0:
                pd2[i].append(checker)
                checker = 0
            if count == "Not Done" and na > 0:
                pd2[str(i)+" skipped"].append(na)
                tot+=na
                na = 0


            if pos == len(new_pb_l):
                if checker!= 0:
                    pd2[i].append(checker)
                    checker=0
                elif na!=0:
                    pd2[str(i)+" skipped"].append(na)
                    na=0

            count=0
            
    return pd2

def new_cate(d = get_data()):
    """
    hot or cold category of numbers
    """
    temp = powerball_list(d)
    pd2 = numbers_skipped_two(temp)
    ne_dict = {}
    for i in range(1,70):
        ne_dict[i]  = sum(pd2[i])
    return ne_dict

    
def pattern (lists, path):
    """
    takes in two list of list and a return a list of list of where th
    """
    new = []
    pos = 0
    while pos < len(lists)-len(path):
        if lists[pos:pos+len(path)] == path:
            new.append(lists[pos+len(path)])
        pos+=1
        
    return new

def next_powerball(pd,pd2):
    """
    takes in two dictionary and return a list of possible outcomes for the next draw
    """
    final = []
    for i in range(1,70):
        verify = ""
        try:
            int(pd[i][-1])
            verify = True
        except:
            verify = False

        if verify:
            l3 = pd2[i][-3:]
            temp = pd2[i]
            temp = sorted(temp)
            if  temp[-5]> pd[i][-1] and temp[-4]> pd[i][-1] and temp[-3]> pd[i][-1] and\
            temp[-2]> pd[i][-1] and temp[-1]> pd[i][-1] and len(pattern(pd2[i], l3)) > 3:
                final.append(i)
        else:
            temp = pd2[str(i)+" skipped"]
            if temp[-1] > max(temp):
                final.append(i)
            else:
                dic = {}
                for j in temp:
                    if j in dic:
                        dic[j] +=1
                    else:
                        dic[j] = 1
                if dic[temp[-1]] >= 2:
                    final.append(i)

    return final

def one_number (n1):
    """
    takes in two a list and return a list of possible outcomes for the next draw
    """
    pd = {}
    pik = {}
    for i in range (1,70):
        pd[i]=[]
        pik[i]=[]

    tot = 0
    count = 0
    checker = 0
    na = 0

    for i in range (1,70):
        pos = 0
        tot= 0
        while pos < len(n1):
            if i == n1[pos]:
                checker+=1
                pos+=1
                count="Not Done"
            else:
                na+=1
                count = "Done"
                pos+=1

            if count == "Done" and checker > 0:
                pd[i].append(checker) 
                tot+=checker
                pik[i].insert(0,checker)
                checker = 0
            if count == "Not Done" and na > 0:
                pd[i].append("skipped ==" +str(na))
                tot+=na
                na = 0
            if pos == len(n1):
                if checker!= 0:
                    pd[i].append(checker)
                    tot+=checker
                    checker=0
                elif na!=0:
                    pd[i].append("skipped ==" +str(na))
                    tot+=na
                    na=0
            count=0
            
    return pd


def one_number_2(n1):

    pd2 = {}

    for i in range (1,70):
        pd2[i]=[]
        pd2[str(i)+" skipped"] = []
    tot = 0
    count = 0
    checker = 0
    na = 0

    for i in range (1,70):
        pos = 0
        tot= 0
        while pos < len(n1):
            if i == n1[pos]:
                checker+=1
                pos+=1
                count="Not Done"
            else:
                na+=1
                count = "Done"
                pos+=1

            if count == "Done" and checker > 0:
                pd2[i].append(checker)
                checker = 0
            if count == "Not Done" and na > 0:
                pd2[str(i)+" skipped"].append(na)
                tot+=na
                na = 0


            if pos == len(n1):
                if checker!= 0:
                    pd2[i].append(checker)
                    checker=0
                elif na!=0:
                    pd2[str(i)+" skipped"].append(na)
                    na=0

            count=0
    return pd2


def red_ball_numbers_skipped (new_pb_l):
    """
    takes in powerball list and returns a dictionary of how many numbers was skipped and
    how many occurence a number has on the red ball
    """
    pd = {}

    for i in range (1,40):
        pd[i]=[]

    tot = 0
    count = 0
    checker = 0
    na = 0

    for i in range (1,40):
        pos = 0
        tot= 0
        while pos < len(new_pb_l):
            if i == new_pb_l[pos]:
                checker+=1
                pos+=1
                count="Not Done"
            else:
                na+=1
                count = "Done"
                pos+=1

            if count == "Done" and checker > 0:
                pd[i].append(checker)
                tot+=checker
                checker = 0
            if count == "Not Done" and na > 0:
                pd[i].append("skipped ==" +str(na))
                tot+=na
                na = 0
            if pos == len(new_pb_l):
                if checker!= 0:
                    pd[i].append(checker)
                    tot+=checker
                    checker=0
                elif na!=0:
                    pd[i].append("skipped ==" +str(na))
                    tot+=na
                    na=0
            count=0
    return pd




def red_ball_numbers_skipped_two(new_pb_l):
    """
    takes in powerball list and returns a dictionary of how many numbers was skipped and
    a list of all occurence a number has on the red ball
    """
    pd2 = {}

    for i in range (1,40):
        pd2[i]=[]
        pd2[str(i)+" skipped"] = []
    tot = 0
    count = 0
    checker = 0
    na = 0

    for i in range (1,40):
        pos = 0
        tot= 0
        while pos < len(new_pb_l):
            if i == new_pb_l[pos]:
                checker+=1
                pos+=1
                count="Not Done"
            else:
                na+=1
                count = "Done"
                pos+=1

            if count == "Done" and checker > 0:
                pd2[i].append(checker)
                checker = 0
            if count == "Not Done" and na > 0:
                pd2[str(i)+" skipped"].append(na)
                tot+=na
                na = 0


            if pos == len(new_pb_l):
                if checker!= 0:
                    pd2[i].append(checker)
                    checker=0
                elif na!=0:
                    pd2[str(i)+" skipped"].append(na)
                    na=0

            count=0
            
    return pd2

def red_ball_next_powerball(pd,pd2):
    """
    takes in two dictionary and return a list of possible outcomes for the next draw for t
    the red ball
    """
    final = []
    for i in range(1,27):
        verify = ""
        try:
            int(pd[i][-1])
            verify = True
        except:
            verify = False

        if verify:
            l3 = pd2[i][-3:]
            temp = pd2[i]
            temp = sorted(temp)
            if temp[-3]> pd[i][-1] and\
            temp[-2]> pd[i][-1] and temp[-1]> pd[i][-1] and len(pattern(pd2[i], l3)) > 3:
                final.append(i)
        else:
            temp = pd2[str(i)+" skipped"]
            if temp[-1] > max(temp):
                final.append(i)
            else:
                dic = {}
                for j in temp:
                    if j in dic:
                        dic[j] +=1
                    else:
                        dic[j] = 1
                if dic[temp[-1]] >= 2:
                    final.append(i)

    return final
##############################################################################
def get_excel():
    year = "2021"
    all_data =  get_data()
    all_data =  get_data()
    all_data.drop([0,1,2,9,10], axis = 1, inplace = True)
    new_data = all_data.loc[all_data[3]== year]
    new_data.drop([3], axis = 1, inplace =True)
    new_data = new_data.values.tolist()
    all_list = []

    for i in new_data:
        for j in i:
            all_list.append(int(j))
    from collections import Counter
    
    all_counts = Counter(all_list)
    m = max(all_counts)
    final = {}
    for i in range(1,m+1):
        final[int(i)] = int(all_counts[i])

    final = pd.DataFrame.from_dict(final, orient = "index")
    final.rename(columns={0: 'Stats'}, inplace= True)
    new_col = [str(i) for i in range(1,m+1)]  
    # can be a list, a Series, an array or a scalar   
    final.insert(loc=0, column='Balls', value=new_col)
    final1 = final.iloc[:m//2, :]
    final2 = final.iloc[m//2:, :]
    def powerball_stats (year):
        year = str(year)
        all_data =  get_data()
        all_data.drop([0,1,2,4,5,6,7,8,10], axis = 1, inplace = True)
        new_data = all_data.loc[all_data[3]== year]
        new_data.drop([3], axis = 1, inplace =True)
        new_data = new_data.values.tolist()
        new_data
        all_list = []

        for i in new_data:
            for j in i:
                all_list.append(int(j))
        from collections import Counter

        all_counts = Counter(all_list)
        m = max(all_counts)
        final = {}
        for i in range(1,m+1):
            final[int(i)] = int(all_counts[i])
        final
        final = pd.DataFrame.from_dict(final, orient = "index")
        final.rename(columns={0: 'Stats'}, inplace= True)
        new_col = [str(i) for i in range(1,m+1)]  # can be a list, a Series, an array or a scalar   
        final.insert(loc=0, column='Balls', value=new_col)
        return final
    pb_stats = powerball_stats(2021)
    # ACCES GOOGLE SHEET
    gc = gspread.service_account(filename='excellent-tide-327402-600be4d1c4b4.json')
    sh = gc.open_by_key('1DJcfAQ8ZEyEUj5y1OT-t1s_eOMD2FV6RjWEfIWBVJZM')

    

    # ACCES GOOGLE SHEET
    credentials = {"type": "service_account",
  "project_id": "excellent-tide-327402",
  "private_key_id": "600be4d1c4b4ad525f41c8a257736d6bef49937c",
  "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvAIBADANBgkqhkiG9w0BAQEFAASCBKYwggSiAgEAAoIBAQCi5234u2l97ypK\n4KcUb/T8balOPKXaRLpsBL/sw9DszMH5WX+Bvwp+0rA/CxqTDnTa1CgGHCTEx7s6\nNe+lG2THbcenHCY98QCrdxoKNEY/+pjQw49/1mLo7H8gHH6xv2bamTb8s/ow0m+o\nTXWJNcU2sVRF2R3jYdeU9HbD88CbCrvt1ayCM71mzFZPbhq1YOkHK5xBFwHFYZ10\nNB7GZZM1ZCNfFqDB6oT9N+J4SbOTzWYpGM45tltmgZgwbq0YYqNTh8S0k7fBCt+v\nsvo4Fo/ArhExuCJ41pCRtn96/lsTnSayFTJn6wBA/lyTyIK+m5l4ZVVZbBk2qHgV\nD58s+sV3AgMBAAECggEAGAYQMjkVMWoRWccv7J7RdAk0tCHT2vtSu2D8ms01Ojln\nYRSL7hWUgH0lWmQSpbLhPxpAKB0uEPUG4pVCA556v2xW9V58h0Wg3H3X0b8WyGhL\nm/yTD35iyh3Xu45oOAyq0ub7HyDoRHCW4IJCffB3kSgX7lBlYiXeNeHSwqASUerI\nN6pJ9NpTaaAaaNIUHn9Tbbuqce93Opq7QndB8bzTn+4vNUG4/hPl2rw9WqFyujM3\nGfV3d57Dcbrr4IOZaqfoAKJjI2BSR9o2L737stPkWoDiHtopBcKBZ2Z6MhPugtX3\ntuiunrIlv9z+Qb7Uq+dA/Yej+0Cs7Xl/aLlKh+AW/QKBgQDUbSyihh8CMigYNROI\no+OjwjEzjdCaaBumfTi6SI+U5QLJwE8gNxGA8WoIxUMPWlfDv9SPDwMTPeKWwcSN\n15Fi3gsujjclYWXCtliyQbSyUAkNJKKgm24pMA0d1s59jCq6mA2yhVyFVVs+l6In\n9XucjdJEfgVnionu0mww2jz3NQKBgQDEUcJU5I6IYHp6sPxXhh3S5eb1M7m36EzF\ncM2jJOQ/W1RGUi7R56aLjdfnkI/OzdY1r/75bTR80QnVF+wOccD77fcAXQCnrs7u\nk9aRWPDgZo0HIXV8EzMJEwFBqit6sRGTwvZX8dQZrvuI2jiX+r4yqCupYqXkvcDX\n1/wFfAzjewKBgGAfEp5sICXnOjtR8QfYWQ5ltcvFNQpZZ4GbkgrBAK94PR27tlI2\neOYm1zsmv6R31dTOvckKGvMfAqQDBATG8ZOSM+8aDRwOCXTk/BeVIcW5746R8EDK\ndDuQ8a5536/xt/f9C92m+OmgaQxWotp/+zIo/mdduuZSSv3VG32zkvitAoGAa/KY\n7J0QUqTaNASFfIrec0y/Biz+/cTaEebI6+ApMT1CxbgCzqCVzz6bbVTr/kbodnTj\nurr1lXxL78RLpFWgS+a5XpXE+m8AkebwoNNK/9jjyvv24dacxWyR0E2Fwt0CUUFu\nu0TZ35V78fAsFkVlt+0ItQbtOS2tkAHPLiTb4WkCgYARTs+k3ODUyHtfgwqzUm85\nc0Vbeci1dYLmSeK6b6c4uVn5uQcmNiA2I5t70LqjCaP8N3AEshIghfonc6je2mQo\nqd8S0N0e1ZmMpl6r9vQ1PxIGc6/rT9Rjp36Sje5nylkxkvPtKjZ//i62Uwp/hSh/\n5ES5Esj02JJruNFA0fXt+A==\n-----END PRIVATE KEY-----\n",
  "client_email": "stats-144@excellent-tide-327402.iam.gserviceaccount.com",
  "client_id": "106236167538551787214",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/stats-144%40excellent-tide-327402.iam.gserviceaccount.com"}

    worksheet = sh.get_worksheet(0)
    worksheet1 = sh.get_worksheet(1)
    worksheet2 = sh.get_worksheet(2)#-> 0 - first sheet, 1 - second sheet etc. 

    # APPEND DATA TO SHEET
    your_dataframe =final1
    your_dataframe1 =final2
    your_dataframe2 =pb_stats
    set_with_dataframe(worksheet, your_dataframe) #-> THIS EXPORTS YOUR DATAFRAME TO THE GOOGLE SHEET
    set_with_dataframe(worksheet1, your_dataframe1)
    set_with_dataframe(worksheet2, your_dataframe2)
#############################################################################################################
# MAIN CODE STARTS HERE
#####################################################################################################33333
def google_pred():
    test_data = get_data()
    red_ball_list = test_data[9].tolist()
    red_ball_list = [int(i) for i in red_ball_list]
    test_data["Date"] = test_data[1].astype(str) +"-"+ test_data[2].astype(str) +"-"+test_data[3].astype(str)
    test_data["num"] = test_data[4].astype(str) +","+ test_data[5].astype(str) +","+test_data[6].astype(str)+","+test_data[7].astype(str)+","+test_data[8].astype(str)
    test_data.drop([0,1,2,3,4,5,6,7,8,9,10], axis = 1, inplace = True)
    test_data_num = test_data["num"].to_list()
    for numbers in range(len(test_data_num)):
    
        num_temp = test_data_num[numbers].split(",")
        #change all to integers
        num_temp[0] =  int(num_temp[0])
        num_temp[1] =  int(num_temp[1])
        num_temp[2] =  int(num_temp[2])
        num_temp[3] =  int(num_temp[3])
        num_temp[4] =  int(num_temp[4])

        num_temp = sorted(num_temp)
        test_data_num[numbers] = num_temp
    test_data["New"]  = test_data_num
    test_data["New"] = test_data["New"].astype(str)
    pb_list = powerball_list()
    k = len(pb_list)-312
    last = -1
    pos = 1150-len(pb_list)
    excel_dict ={"Date":[], "Powerball Prediction":[], "Actual Numbers":[], "No. Of Correct":[],
                 "Red Ball Prediction":[], "Actual Red Ball":[]}
    corr = 0
    while last > pos-1:
        pb_list_temp = pb_list[k:last]
        rb_temp = red_ball_list[k:last]

        rb = red_ball_numbers_skipped(rb_temp)
        rb2 = red_ball_numbers_skipped_two(rb_temp) 

        pd_temp = numbers_skipped(pb_list_temp)
        pd2_temp = numbers_skipped_two(pb_list_temp)
        final_temp = next_powerball(pd_temp,pd2_temp)
        red_ball_final = red_ball_next_powerball(rb, rb2)
        excel_dict["Powerball Prediction"].append(final_temp)
        excel_dict["Actual Numbers"].append(pb_list[last])
        excel_dict["Date"].append(test_data.loc[test_data['New'] == str(pb_list[last]), ['Date']].values[0][0])
#         excel_dict["No. Of Correct"].append(sum(el in pb_list[last] for el in final_temp))
        excel_dict["Red Ball Prediction"].append(red_ball_final)
        excel_dict["Actual Red Ball"].append(red_ball_list[last])
        last-=1
    
    
    
    last_date = excel_dict["Date"][0]
    last_date = last_date.split("-")
    
    last_date1 = date(day=int(last_date[1]), month=int(last_date[0]), 
         year=int(last_date[2])).strftime('%A %d %B %Y')

    word_date = last_date1.split(" ")
    day = 0
    if word_date[0] == "Monday":
        day = 2
    elif word_date[0] == 'Wednesday':
        day = 3
    elif word_date[0] == "Saturday":
        day = 2
    month=array.array('i',[0,31,28,31,30,31,30,31,31,30,31,30,31])
    count=0
    d = int(last_date[1])
    months = int(last_date[0])
    year = int(last_date[2])
    days = day

    if year%4==0:
        month[2]=28
    while count<days:
        d=d+1
        count=count+1
        if d>month[months]:
            d=1
            months=months+1
        if months>12:
            months=1
            year=year+1
            if year%4==0:
                month[2]=29
            else:
                month[2]=28

    future = str(str(months)+"-"+str(d)+"-"+str(year))    
    ##############################################################
    nx_d = future.split("-")
    mm , dd, yy = int(nx_d[0]),int(nx_d[1]),int(nx_d[2])

    d= get_data()
    d1 = d.iloc[312:] 
    d1.reset_index(drop=True, inplace=True)
    logreg = Prediction(yy,mm,dd, kind = "l",data= d1)
    dtree = Prediction(yy,mm,dd, kind = "d",data = d1)

    new_log = list(logreg.values())
    new_tree = list(dtree.values())
    new_log = [new_log[0],new_log[3], new_log[4]]
    new_tree = [new_tree[0], new_tree[3], new_tree[4]]
    new_final = list(set(new_log + new_tree))
    l = powerball_list(d1,kind = "s")
    l = l[-10:]
    def mc_final (l):
        n1 = []
        n2 = []
        n3 = []

        for i in l:
            n1.append(i[0])
            n2.append(i[3])
            n3.append(i[4])
        l = list(set(n1+n2+n3))
        return l
    t = mc_final(l)
    pb_list = powerball_list()
    k = len(pb_list)
    last = len(pb_list)-50
    k = len(pb_list)-312
    last = -1
    pos = 1150-len(pb_list)
    nc= []

    corr = 0
    while last > pos-1:
        pb_list_temp = pb_list[k:last]
        pd_temp = numbers_skipped(pb_list_temp)
        pd2_temp = numbers_skipped_two(pb_list_temp)
        final_temp = next_powerball(pd_temp,pd2_temp)
        final_temp = list(set(final_temp+t))
        nc.append(sum(el in pb_list[last] for el in final_temp))
        last-=1
        
    excel_dict["No. Of Correct"] = nc
    ##########################################################
    
    
    start = len(pb_list)-312
    pb_list_future = pb_list[start:]
    pd_future = numbers_skipped(pb_list_future)
    pd2_future = numbers_skipped_two(pb_list_future)
    future_pred = next_powerball(pd_future,pd2_future)
    future_pred = list(set(future_pred+ t))
    
    rb_list_future = red_ball_list[start:]
    rb_future = red_ball_numbers_skipped(rb_list_future)
    rb2_future = red_ball_numbers_skipped_two(rb_list_future)
    rb_future_pred = red_ball_next_powerball(rb_future,rb2_future)
    
    future_actual =""
    future_correct =""
    excel_dict["Date"].insert(0,future)
    
    excel_dict["Powerball Prediction"].insert(0,future_pred)
    excel_dict["Actual Numbers"].insert(0,future_actual)
    excel_dict["No. Of Correct"].insert(0,future_correct)
    excel_dict["Red Ball Prediction"].insert(0,rb_future_pred)
    excel_dict["Actual Red Ball"].insert(0,future_actual)
    
    
    
    final_df = pd.DataFrame(excel_dict)
    final_df["Actual Numbers"] = final_df["Actual Numbers"].astype(str)
    final_df["Powerball Prediction"] = final_df["Powerball Prediction"].astype(str)
    final_df["Red Ball Prediction"] = final_df["Red Ball Prediction"].astype(str)
    def apply (dt):
        dt = dt.replace("[","")
        dt = dt.replace("]","")
        return dt
    
    final_df["Actual Numbers"] = final_df["Actual Numbers"].apply(apply)
    final_df["Powerball Prediction"] = final_df["Powerball Prediction"].apply(apply)
    final_df["Red Ball Prediction"] = final_df["Red Ball Prediction"].apply(apply)
    
    def break_down(x):
        x = x.split(",")
        pos =  0 
        while len(x)> pos:
            pos+=7
            x.insert(pos,"\n")
        x = str(apply(str(x)))
        return x
    def break_down2(x):
        x = x.split(",")
        fin = ",".join(x[0:7])
        start = 7
        pos =  14
        while len(x)+8> pos:
            temp = ",".join(x[start:pos])
            start = pos
            pos+=7

            fin = fin + "\n" + temp
        return fin
    
    final_df["Powerball Prediction"] = final_df["Powerball Prediction"].apply(break_down2)
    final_df["Red Ball Prediction"] = final_df["Red Ball Prediction"].apply(break_down2)
    ########################################################
    # WRITING TO GOOGLE EXCEL SHEET
    #########################################333
    credentials = {"type": "service_account",
  "project_id": "imposing-pipe-326023",
  "private_key_id": "a59a52592ae3e32db333f79fad088060f9f84ce5",
  "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvAIBADANBgkqhkiG9w0BAQEFAASCBKYwggSiAgEAAoIBAQC5A8LMy7uppomC\nxaizCzzN4SlFgC1Gi9bJllUcYtoUfT0aCi7JiBPamZNLPiCPxf3LRAZmNYBLpuAi\nLalFrE6qcK+N3b5OXNRnCkH+gTeAkygPr/HawpQpH/HGfyC7YpOB//tLSSGE31cl\nnuikyXyfga76kWNdDvxwqEUEXKOEv9Z+0cohmCAfvjEwWWZ5vBZ+ie8yAzInMRc1\nVAr1ByFzha5fekTfiqHX86NATVYz3YgthO4hSzMMLpQZWmiEfDnRD2DhC7GmvkE0\ndTc4Z/AHQFI5He5p9YFDyYeuvIZKaEIG9RxLThtneuiI1eglHPvKZ2pVH49hbQ1p\nBrFe/w/VAgMBAAECggEAKclxq/Ov9QdKM6EVEytMlmtueOYUU8StxGYR3xFslAgd\nTs5R9u6nHU5meC4WCKL9SXWZxGf9DBkqsk2B19ys/83nuLEGlIhe6M7mTOc+c+nI\nuJJSm8gq0ocGgoCgEfKXwlrglQZ1bZ9L/ZFAlkdzwEr4DFveB1ylI4S94dhSjl6q\nuHPonEoqlU9sHdiTSp2Avufbyr5B3qCRVsgq8rLut8XvYWwfrfqI/xxchz/1b9f2\nmQfVRXPJYtCsjTAsmcbsAsoJNzBFAYklv0jPtGDr2pTfDZ/TNrvrQ417OcrJrveo\nOuGN9JvZrsL05q52Rl5zgS/RZ3RIc6wp5JVDS3JlwQKBgQD1vqh9VGJkztPzDhvH\n4gkwWufKvkRypPNosTsk2Pri6ASFzW08Sd9SWKhWnWBXDqmpR5ut6P54OIvdgCN2\nX2t/soOSgBbQMaRhXfbHvS/EwpD3c7fe/8GCcOVNjoLQoAsqhN4nSDFy6hyI3LOc\ndno0UWSeRMDvrIQEvj4CAvz5zQKBgQDAvE+bdKVxKhIH8aobwL5KRRsgOMS4tr2q\nPPROsconLz3uKMSMcIzsePDRgMbvlVGlHaDHngKohKpwEnCTis10C44xWpnWLzoO\nLk7bc1J0DdjwHy3NcTSMg2QXr9ew3Jl/7QLG1PARFD0jfgCE8t6EprdG8J8Z4QlQ\nUxQX1+FGKQKBgAj0TYdjj8JElwyAMxrxbYxJg7Crhir3P7dM3e7VyS6DbcbCWXyc\n5HpHqLqfOWdyrVPxvAZ2Ou9+f/ouNRyXFX+trYWlDS/A31B88AUPK7JrtISPvt9t\nSkvKeVB+JN4dNsrx1HZx9vtM9IU4JYNJ/eHyJUxvDOiuzJCvreq82SLxAoGATfKa\n24tEcc0K982D969vBkiGnPR4kx/M+zGluMHsuQZBLLWuMAJA7E5JTuGfEzw3hejC\nopjECHWHHMZIY1NqnjkiK9Gxj88P0rZlzBkKysbi6tIhSwoyr3VgILhMKko9hmBL\nlDCAWtftlhIakapL1ig6zWT5Y5UAmEzRPodfo2kCgYAfEw3pKIrLosT82zMwEN+S\nabB11yJPUHtoKb7ruloTH2sfvL5BqFSkDNeP0/suoAXHkNGfvgn41/cn1RIMOhvK\nKkVqowgqTG/gItkmUuHQsXVsuG5hr6IUfEARkUMJGJiaqppBFNeQ+pzMRuyd3CDt\nsiK98KYzAE/KFhy/tmbtTQ==\n-----END PRIVATE KEY-----\n",
  "client_email": "d3east@imposing-pipe-326023.iam.gserviceaccount.com",
  "client_id": "107751742785254946026",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/d3east%40imposing-pipe-326023.iam.gserviceaccount.com"}

    gc = gspread.service_account_from_dict(credentials)

    sh = gc.open_by_key('1K7jbaMTh7SBEEoVRcz5UuX_VqLkd4FnmSMPIoDmn7-E')

    worksheet = sh.get_worksheet(0) #-> 0 - first sheet, 1 - second sheet etc. 

    # APPEND DATA TO SHEET
    your_dataframe =final_df
    set_with_dataframe(worksheet, your_dataframe)
    get_excel()
    return day

        
while True:
    d = google_pred()
    time.sleep(d*24*60*60)
    
        
