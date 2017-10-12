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
ydata2=[]
ydata3=[]
CO=[]
AMB_TEMP=[]
CH4=[]
NMHC=[]
NO=[]
NO2=[]
NOx=[]
O3=[]
PM10=[]
RAINFALL=[]
RH=[] 
SO2=[] 
THC=[] 
WD_HR=[] 
WIND_DIREC=[] 
WIND_SPEED=[] 
WS_HR=[]


#read pm2.5 in line,ydata
line_number = 0
for line in csv.reader(open("train.csv",'r')):
	if (line_number-1)%18 == 0:
		del line[0:3]
		AMB_TEMP = AMB_TEMP +line
	if (line_number-1)%18 == 1:
		del line[0:3]
		CH4 = CH4 +line
	if (line_number-1)%18 == 2:
		del line[0:3]
		CO = CO +line
	if (line_number-1)%18 == 3:
		del line[0:3]
		NMHC = NMHC +line
	if (line_number-1)%18 == 4:
		del line[0:3]
		NO = NO +line
	if (line_number-1)%18 == 5:
		del line[0:3]
		NO2 = NO2 +line
	if (line_number-1)%18 == 6:
		del line[0:3]
		NOx = NOx +line
	if (line_number-1)%18 == 7:
		del line[0:3]
		O3 = O3 +line
	if (line_number-1)%18 == 8:
		del line[0:3]
		PM10 = PM10 +line
	if (line_number-1)%18 == 9:
		del line[0:3]
		ydata = ydata +line
		ydata2= ydata2 +line
		ydata3= ydata3 +line
	if (line_number-1)%18 == 10:
		del line[0:3]
		RAINFALL = RAINFALL +line
	if (line_number-1)%18 == 11:
		del line[0:3]
		RH= RH +line
	if (line_number-1)%18 == 12:
		del line[0:3]
		SO2 = SO2 +line
	if (line_number-1)%18 == 13:
		del line[0:3]
		THC = THC +line
	if (line_number-1)%18 == 14:
		del line[0:3]
		WD_HR = WD_HR +line
	if (line_number-1)%18 == 15:
		del line[0:3]
		WIND_DIREC = WIND_DIREC +line
	if (line_number-1)%18 == 16:
		del line[0:3]
		WIND_SPEED = WIND_SPEED +line
	if (line_number-1)%18 == 17:
		del line[0:3]
		WS_HR = WS_HR+line
	line_number +=1

ydata2= np.asarray(ydata2)
ydata2 = ydata2.astype(np.float)
ydata3= np.asarray(ydata3	)
ydata3 = ydata2.astype(np.float)
def square(list):
	ret=[]
	for i in list:
		ret.append(i*i)
	return ret
'''	
def cube(list):
	ret3=[]
	for i in list:
		ret3.append(i*i*i)
	return ret3
'''

ysqr=square(ydata2)
#ycube=square(ydata3)


#print(ysqr)


x_item =[]  #first 5 hr pm2.5
y_item =[]	#6th hr pm2.5
co_item=[]  #CO
feature_number = 5  #how many hour picked up
hi=17	#want how many features
for j in range(0, len(ydata) - feature_number):
	if (j%480)>=0 and (j%480)<=(479-feature_number):  #bcuz source data only contains first 20days of one month,must seperate 20th day and first day of next month
		x_item.append(ydata[j : j + feature_number]+AMB_TEMP[j : j + feature_number]+CH4[j : j + feature_number]+CO[j : j + feature_number]+
			NMHC[j : j + feature_number]+NO[j : j + feature_number]+NO2[j : j + feature_number]+NOx[j : j + feature_number]
			+O3[j : j + feature_number]+PM10[j : j + feature_number]+RH[j : j + feature_number]
			+SO2[j : j + feature_number]+THC[j : j + feature_number]+WD_HR[j : j + feature_number]+WIND_DIREC[j : j + feature_number]
			+WIND_SPEED[j : j + feature_number]+WS_HR[j : j + feature_number]+[1])
		#x_item.append(['1'])
		#+NO2[j : j + feature_number]+NOx[j : j + feature_number]+SO2[j : j + feature_number]+PM10[j : j + feature_number]
		'''
		AMB_TEMP[j : j + feature_number]+CH4[j : j + feature_number]+CO[j : j + feature_number]+
			NMHC[j : j + feature_number]+NO[j : j + feature_number]+NO2[j : j + feature_number]+NOx[j : j + feature_number]
			+O3[j : j + feature_number]+PM10[j : j + feature_number]+RH[j : j + feature_number]
			+SO2[j : j + feature_number]+THC[j : j + feature_number]+WD_HR[j : j + feature_number]+WIND_DIREC[j : j + feature_number]
			+WIND_SPEED[j : j + feature_number]+WS_HR[j : j + feature_number]
		'''
		y_item.append(ydata[j + feature_number])
##add
##+CO[j : j + feature_number]

		
x_item = np.asarray(x_item)
x_item = x_item.astype(np.float)
y_item = np.asarray(y_item)
y_item = y_item.astype(np.float)

#print(ydata)
#print(len(x_item))
#print(len(ydata))




    # start training w

time = 15000
l_rate = 15
w = [1] * ((hi*feature_number)+1)	
data = len(y_item)

for i in range(0,time):
	total = [0] * (feature_number+1)
	cost= 0

x_t = x_item.transpose()
s_gra = np.zeros(len(x_item[0]))
#s_b_gra = 0
#bias=0
#bias = np.zeros((len(x_item)))
for i in range(time):
    hypo = np.dot(x_item,w)
    #loss = y_item -hypo
    loss = (hypo - y_item)
    cost = np.sum(loss**2) / len(x_item)
    cost_a  = math.sqrt(cost)
    gra = np.dot(x_t,loss)
    s_gra += gra**2
    ada = np.sqrt(s_gra)
    #b_gra = loss[feature_number]
    #b_gra = np.dot(np.ones(((len(x_item)))),loss)
    #s_b_gra += b_gra**2 
    #b_ada = np.sqrt(s_b_gra)
    w = w - l_rate * gra/ada
    #bias = bias + l_rate * b_gra/ada[feature_number]
    print ('iteration: %d | Cost: %f  ' % ( i,cost_a))
    #print(loss)

	
#print(cost/(2*data))

for i in range(0,(hi*feature_number+1)):
	print(w[i],",")   #bias is the last term
#print(bias,'bias')
