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

#read last 5 hrs pm2.5 in line,x_in
featurenum=9 #how many hrs 
line_number = 0
for line in csv.reader(open("test.csv",'r')):
	if (line_number)%18 == 4:
		del line[0:(11-featurenum)]
		NO = NO +line
	if (line_number)%18 == 5:
		del line[0:(11-featurenum)]
		NO2= NO2 +line
	if (line_number)%18 == 6:
		del line[0:(11-featurenum)]
		NOx = NOx +line
	if (line_number)%18 == 12:
		del line[0:(11-featurenum)]
		SO2 = SO2 +line
	if (line_number)%18 == 8:
		del line[0:(11-featurenum)]
		PM10 = PM10 +line
	if (line_number)%18 == 9:
		del line[0:(11-featurenum)]
		x_in = x_in +line   #add pm2.5 into list
		x2 = x2+line
	if (line_number)%18 == 2:
		del line[0:(11-featurenum)]
		CO = CO +line #add co into list
	line_number +=1
x2= np.asarray(x2)
x2 = x2.astype(np.float)
def square(list):
	ret=[]
	for i in list:
		ret.append(i*i)
	return ret


pmsqr=square(x2)



hi=2  #terms add
x_item=[]
for j in range(0, len(x_in),featurenum):
	if (j%featurenum)>=0 and (j%featurenum)<=(featurenum-1):  #bcuz source data only contains first 20days of one month,must seperate 20th day and first day of next month
		x_item.append(x_in[j : j + featurenum]+pmsqr[j : j + featurenum]+[1])
		#+co_in[j : j + featurenum] #add CO
		#NO[j : j + featurenum]+NO2[j : j + featurenum]+NOx[j : j + featurenum]+SO2[j : j + featurenum]+PM10[j : j + featurenum]
		
x_item = np.asarray(x_item)
x_item = x_item.astype(np.float)


index=0

w=[0.0377636956056 ,
-0.00166884689462 ,
0.0128472677869 ,
-0.00660918315642 ,
0.0265180945013 ,
0.0748740768055 ,
-0.037071062883 ,
0.214578824965 ,
0.644472555368 ,
-0.000963737522223 ,
-4.30450942429e-06 ,
0.00187216886302 ,
-0.00232790106231 ,
-0.000487112363781 ,
0.00459101022611 ,
-0.00640408178186 ,
-0.00215444842767 ,
0.00534807984035 ,
0.949106419935] 
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
