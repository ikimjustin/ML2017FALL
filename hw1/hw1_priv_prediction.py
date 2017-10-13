import sys
import csv
import pandas as pd
import numpy as np 


x_in=[] #pm2.5
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
x2=[]
x3=[]

#read last 5 hrs pm2.5 in line,x_in
feature_number=9 #how many hrs
featurenum=9
line_number = 0
for line in csv.reader(open("test.csv",'r')):
	if (line_number)%18 == 0:
		del line[0:(11-featurenum)]
		AMB_TEMP = AMB_TEMP +line
	if (line_number)%18 == 1:
		del line[0:(11-featurenum)]
		CH4 = CH4 +line
	if (line_number)%18 == 2:
		del line[0:(11-featurenum)]
		CO = CO +line #add co into list
	if (line_number)%18 == 3:
		del line[0:(11-featurenum)]
		NMHC = NMHC +line
	if (line_number)%18 == 4:
		del line[0:(11-featurenum)]
		NO = NO +line
	if (line_number)%18 == 5:
		del line[0:(11-featurenum)]
		NO2= NO2 +line
	if (line_number)%18 == 6:
		del line[0:(11-featurenum)]
		NOx = NOx +line
	if (line_number)%18 == 7:
		del line[0:(11-featurenum)]
		O3 = NOx +line
	if (line_number)%18 == 8:
		del line[0:(11-featurenum)]
		PM10 = PM10 +line
	if (line_number)%18 == 9:
		del line[0:(11-featurenum)]
		x_in = x_in +line   #add pm2.5 into list
		x2 = x2+line
		x3= x3=line
	if (line_number)%18 == 11:
		del line[0:(11-featurenum)]
		RH = RH +line
	if (line_number)%18 == 12:
		del line[0:(11-featurenum)]
		SO2 = SO2 +line
	if (line_number)%18 == 13:
		del line[0:(11-featurenum)]
		THC = THC +line
	if (line_number)%18 == 14:
		del line[0:(11-featurenum)]
		WD_HR = WD_HR +line
	if (line_number)%18 == 15:
		del line[0:(11-featurenum)]
		WIND_DIREC = WIND_DIREC +line
	if (line_number)%18 == 16:
		del line[0:(11-featurenum)]
		WIND_SPEED = WIND_SPEED +line
	if (line_number)%18 == 17:
		del line[0:(11-featurenum)]
		WS_HR = WS_HR +line
	
	line_number +=1
x2= np.asarray(x2)
x2 = x2.astype(np.float)
x3= np.asarray(x3)
x3 = x3.astype(np.float)
def square(list):
	ret=[]
	for i in list:
		ret.append(i*i)
	return ret
'''
def cube(list):
	ret=[]
	for i in list:
		ret.append(i*i*i)
	return ret
'''

pmsqr=square(x2)
#pmcube=cube(x3)



hi=1  #terms add
x_item=[]
for j in range(0, len(x_in),feature_number):
	if (j%feature_number)>=0 and (j%feature_number)<=(feature_number-1):  #bcuz source data only contains first 20days of one month,must seperate 20th day and first day of next month
		x_item.append(x_in[j : j + feature_number]+[1])
		#+co_in[j : j + featurenum] #add CO
		#+pmsqr[j : j + feature_number]
		#NO[j : j + featurenum]+NO2[j : j + featurenum]+NOx[j : j + featurenum]+SO2[j : j + featurenum]+PM10[j : j + featurenum]
		'''
		+AMB_TEMP[j : j + feature_number]+CH4[j : j + feature_number]+CO[j : j + feature_number]+
			NMHC[j : j + feature_number]+NO[j : j + feature_number]+NO2[j : j + feature_number]+NOx[j : j + feature_number]
			+O3[j : j + feature_number]+PM10[j : j + feature_number]+RH[j : j + feature_number]
			+SO2[j : j + feature_number]+THC[j : j + feature_number]+WD_HR[j : j + feature_number]+WIND_DIREC[j : j + feature_number]
			+WIND_SPEED[j : j + feature_number]+WS_HR[j : j + feature_number]
		'''
		
x_item = np.asarray(x_item)
x_item = x_item.astype(np.float)


index=0

w=[-0.0331884901913 ,
-0.02592344269 ,
0.218120968139 ,
-0.239630318395 ,
-0.0545005497888 ,
0.530836613079 ,
-0.572080880077 ,
0.000798460274677 ,
1.09459558517 ,
1.73370058295] 
print(w[hi*featurenum])


file=open("result.csv",'w')
csv.writer(file).writerow(['id','value'])
    
for i in range(0,len(x_item)):
	id_string = 'id_' + str(i)
   
	#inner product
	y=0
	y=np.dot(w,x_item[i])
	'''
	for j in range(0,(featurenum*hi)):  #both co pm2.5 added >>times 2
		y+=w[j]*x_item[i][j]
	y+=w[hi*featurenum]
	'''
	csv.writer(file).writerow([id_string,y])

file.close()
