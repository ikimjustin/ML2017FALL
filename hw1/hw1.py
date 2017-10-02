import sys
import csv
import pandas as pd
import numpy as np 

#read file

#data=pd.read_csv("train.csv")
#pd.read_csv(r"train.csv",usecols=[2])
#print (data)

#data['title']
#data['title'].values
row=list()
xdata=list()
ydata=list()
for i in range (3,27):
	row = pd.read_csv("train.csv",sep=',',usecols=[i]).values.tolist()  ##"need .values.tolist" in order to transfer dataframe to list datatype ,so u can op in list
	
	#print (pd.read_csv("train.csv",sep=',',usecols=[i]))
		#for j in range(0,len(row[0])):
	#print (row)	
	for j in range(0,len(row),18):
		xdata.append(row[j:j+9]+row[j+10:j+18])
		ydata.append(row[j+9:j+10])
		
		#print(xdata[j])


print(xdata[0])
print(ydata[0])
print(len(xdata))


#df.5 = df.5.astype(float).fillna(0.0)
