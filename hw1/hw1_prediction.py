import sys
import csv
import pandas as pd
import numpy as np 


x_in=[]

#read last 5 hrs pm2.5 in line,x_in
line_number = 0
for line in csv.reader(open("test.csv",'r')):
	if (line_number)%18 == 9:
		del line[0:2]
		x_in = x_in +line 
	line_number +=1


x_item=[]
for j in range(0, len(x_in),9):
	if (j%9)>=0 and (j%9)<=8:  #bcuz source data only contains first 20days of one month,must seperate 20th day and first day of next month
		x_item.append(x_in[j : j + 9])
		
x_item = np.asarray(x_item)
x_item = x_item.astype(np.float)


index=0

w=[-3.31978887e-02 , -2.59015465e-02 ,  2.18104696e-01 , -2.39639361e-01,
  -5.44710466e-02  , 5.30814168e-01 , -5.72083927e-01,   8.17125089e-04,
   1.09458464e+00, 1.73372888 ] 


file=open("result.csv",'w')
csv.writer(file).writerow(['id','value'])
    
for i in range(0,len(x_item)):
	id_string = 'id_' + str(i)
   
	#inner product
	y=0
	for j in range(0,9):
		y+=w[j]*x_item[i][j]
	y+=w[5]
	csv.writer(file).writerow([id_string,y])

file.close()
