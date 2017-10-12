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
feature_number=5 #how many hrs
featurenum=5
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



hi=17  #terms add
x_item=[]
for j in range(0, len(x_in),feature_number):
	if (j%feature_number)>=0 and (j%feature_number)<=(feature_number-1):  #bcuz source data only contains first 20days of one month,must seperate 20th day and first day of next month
		x_item.append(x_in[j : j + feature_number]+AMB_TEMP[j : j + feature_number]+CH4[j : j + feature_number]+CO[j : j + feature_number]+
			NMHC[j : j + feature_number]+NO[j : j + feature_number]+NO2[j : j + feature_number]+NOx[j : j + feature_number]
			+O3[j : j + feature_number]+PM10[j : j + feature_number]+RH[j : j + feature_number]
			+SO2[j : j + feature_number]+THC[j : j + feature_number]+WD_HR[j : j + feature_number]+WIND_DIREC[j : j + feature_number]
			+WIND_SPEED[j : j + feature_number]+WS_HR[j : j + feature_number]+[1])
		#+co_in[j : j + featurenum] #add CO
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

w=[-0.0377383006024 ,
0.347093678941 ,
-0.437307807208 ,
0.0271031523629 ,
0.916652016255 ,
-0.0227678390802 ,
-0.0163864581682 ,
-0.00717570764305 ,
-0.0071445585554 ,
0.0401311893071 ,
0.0878102017324 ,
0.116583813711 ,
0.101626890157 ,
0.112965739902 ,
0.18186662029 ,
0.148428429506 ,
-0.0893654372709 ,
-0.187759297211 ,
0.297779322373 ,
0.818674071539 ,
0.172578360392 ,
-0.064424228049 ,
-0.0864639208761 ,
-0.0106874340728 ,
0.277107961136 ,
0.0303511458612 ,
-0.0693674375496 ,
0.0820216907919 ,
0.0811398598943 ,
-0.124380856124 ,
-0.0300533605807 ,
0.00330917673877 ,
-0.069025461609 ,
-0.0539516552575 ,
0.164720303817 ,
-0.00425301050611 ,
-0.00219437618499 ,
-0.0390585478549 ,
-0.00202452404321 ,
0.118772074545 ,
-0.0156537306007 ,
-0.0241428370743 ,
-0.0308741228996 ,
-0.00676557460528 ,
0.104195146174 ,
-0.00314467934757 ,
-0.00992698732772 ,
0.0178710179874 ,
-0.0085527884197 ,
0.0504438731152 ,
-0.0191724554 ,
-0.0024364423624 ,
-0.023233741744 ,
0.00569869942513 ,
0.0113493461295 ,
-0.0262994897026 ,
0.0508169687405 ,
-0.0813176362164 ,
0.0648406498151 ,
0.198544303856 ,
0.0974138701958 ,
0.112662046849 ,
0.0877248193207 ,
0.0922658399023 ,
0.175948186722 ,
0.000246268427404 ,
0.00174924252639 ,
-0.00251938879951 ,
0.00109988009266 ,
0.000441029992136 ,
0.000676578291053 ,
0.000204626433149 ,
-5.83914021342e-05 ,
-0.00263104528461 ,
0.000386011974962 ,
-0.0291627241707 ,
-0.0172955959693 ,
-0.0897210652859 ,
-0.104596553999 ,
-0.0274602012625 ,
0.0155338547729 ,
0.128338661424 ,
-0.0734614824938 ,
0.0401713933724 ,
-0.15203538494 ,
0.0809800747296] 
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
