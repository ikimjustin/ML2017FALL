import sys
import csv
#import pandas as pd
import numpy as np 
import math

#read file

#data=pd.read_csv("train.csv")
#pd.read_csv(r"train.csv",usecols=[2])
#print (data)

#data['title']
#data['title'].values
row=()
xdata=[]
ydata=[]

#read pm2.5 in line,ydata
line_number = 0
for line in csv.reader(open("train.csv",'r')):
	if (line_number-1)%18 == 9:
		del line[0:3]
		ydata = ydata +line 
	line_number +=1

'''
for i in range (3,27):
	row = pd.read_csv("train.csv",sep=',',usecols=[i]).values.tolist()  ##"need .values.tolist" in order to transfer dataframe to list datatype ,so u can op in list
	print (row)
	#print (pd.read_csv("train.csv",sep=',',usecols=[i]))
		#for j in range(0,len(row[0])):
	#print (row)	
	for j in range(0,len(row),18):
		#temp = row[j:j+9]+row[j+10:j+18]   #other data
		#temp = np.asarray(temp)
		#temp = temp.astype(np.float)
		#xdata.append(temp)
		temp2 = row[j+9]   #pm2.5
		temp2 = np.asarray(temp2)
		temp2 = temp2.astype(np.float)
		temp2 = temp2.tolist()
		ydata = ydata + temp2
'''


x_item =[]  #first 5 hr pm2.5
y_item =[]	#6th hr pm2.5

feature_number = 9	
for j in range(0, len(ydata) - feature_number):
	if (j%480)>=0 and (j%480)<=(479-feature_number):  #bcuz source data only contains first 20days of one month,must seperate 20th day and first day of next month
		x_item.append(ydata[j : j + feature_number])
		#x_item.append([1])
		y_item.append(ydata[j + feature_number])
x_item = np.asarray(x_item)
x_item = x_item.astype(np.float)
y_item = np.asarray(y_item)
y_item = y_item.astype(np.float)

#print(ydata)
#print(len(x_item))
#print(len(ydata))

'''
#df.5 = df.5.astype(float).fillna(0.0)
def countdiff(base,weight,train):
    group_number=len(train[0])
    #data=[(i[0],i[1],i[2])for i in train]
    group=0
    diff_w=numpy.zeros((5),float)
    diff_b=0
    while group<group_number-3:
        data=[[i[group],i[group+1],i[group+2]] for i in train ]
        temp=data*weight
        temp1=[[sum(i)]for i in temp]
        temp2=sum(map(sum,temp1))
        temp3=train[9][group+3]-temp2-base
        #data=numpy.asarray(data).reshape(54)
        diff_w=diff_w+2*temp3*data*(-1)
        diff_b=diff_b+temp3
        if group%24==20:
            group+=4
        else:
            group+=1
    return [diff_b,diff_w]
'''

    # start training w

time = 10000
l_rate = 0.8
w = [1] * (feature_number )	
data = len(y_item)
'''
for i in range(0,time):
	total = [0] * (feature_number+1)
	cost= 0
'''
x_t = x_item.transpose()
s_gra = np.zeros(len(x_item[0]))
s_b_gra = 0
bias = np.ones((len(x_item)))
for i in range(time):
    hypo = np.dot(x_item,w)+bias
    loss = hypo - y_item
    cost = np.sum(loss**2) / len(x_item)
    cost_a  = math.sqrt(cost)
    gra = np.dot(x_t,loss)
    s_gra += gra**2
    ada = np.sqrt(s_gra)
    b_gra = np.dot(np.ones((1,(len(x_item)))),loss)
    s_b_gra += b_gra**2 
    b_ada = np.sqrt(s_b_gra)
    w = w - l_rate * gra/ada
    bias = bias - b_gra*l_rate/b_ada
    print ('iteration: %d | Cost: %f  ' % ( i,cost_a))


'''
	for j in range(0,data):
		# get wx by using inner product
		inner_product = 0
		for k in range(0,(feature_number+1)):
			inner_product += w[k] * (x_item[j][k])		
		# get y - wx
		inner_product -= (y_item[j])		
		# get total cost
		cost = cost + inner_product ** 2		
		# get Σ(y - wx)x 
		for k in range(0,feature_number):
			total[k] += learning_rate * inner_product * (x_item[j][k])			
		# constant
		total[5] += learning_rate * inner_product			
	# update w
	for k in range(0,(feature_number+1)):
		w[k] = w[k] - total[k] / data	
	if (i%100) == 1:
		print (cost)
'''
		
print(cost/(2*data))
print(w,'%f')
print(bias)
