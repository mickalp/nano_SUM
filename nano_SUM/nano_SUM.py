#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 13:57:02 2023

@author: admin
"""

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import math
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import itertools
from simple_colors import *



path = input('Path to your file and name of the file in xlsx format:'\
             '\n Example: Users/admin/dane.xlsx'\
                 '\n Depends of your system version slash can be in the opposite site '\
                     '\n NOW PLEASE TYPE YOUR PATH:  ')
sheet_name = input('NAME OF SHEET NAME:  ')
df = pd.read_excel(path, sheet_name=sheet_name) #Mac verison



df = df.iloc[:, [0, 1, 2]]
name_column = (df.iloc[:, [0]])
res = name_column.nunique()
fit_str = str('FITTING')
print(f'THIS IS {fit_str} FOR: {sheet_name}')



add_number = int(len(name_column)/res[0])
starts1 = 0-add_number
starts2 = 0

z = add_number
all_y = []

popt_Kalapus = []

logaritmic_r2 = []
logarimtic_r2_scipy = []
logarimtic_r2_numpy = []

poly_r2_scipy = []
lin_r2_scipy = []

coeff_log = []

r2_LG_list = []
exp_list = []
y_kal = []
y_exp_list_test = []
popt_exp = []




def LG_fit(x, y):

    x_data = np.asarray(x)
    y_data = np.asarray(y)

    y_final = max(y)
    y_0 = y[0]
    diff_yf_y0 = abs(y_final-y_0)

    def new_LG(x, A, k, B):
        if y_final > 0:
            y = (y_final*(B-np.exp(-k*x)))*A
        elif y_final < 0 and y_final > y_0:

            y = (y_final*(B-np.exp(-k*(1/x))))*A
        else:
            y = (y_final*(B-np.exp(-k*x)))*A

        return y



    if y_final > 0:
        bounds_range = (0, [5, 2, 30.])
    elif y_final < 0 and y_final > y_0 and not diff_yf_y0 > 3:
        bounds_range = (-80, [20., 20., 5.])
    elif (y_final < 0) and (y_final > y_0) and (diff_yf_y0 > 1):
        bounds_range = (0, [2, 2, 3])
    else:
        bounds_range = (0, [0.3, 0.6, 42.])

    try:

        popt, pcov = curve_fit(new_LG, x_data, y_data, bounds=bounds_range)


        r2_new_LG = r2_score(y_data, new_LG(x_data, *popt))
        r2_LG_list.append(r2_new_LG)
        y_kal.append(new_LG(x_data, *popt))
        print('Kalapus R2: ', round(r2_new_LG, 3))

        popt_Kalapus.append(popt)
        print('Kalapus parameters: 0(theta)=%5.3f, k=%5.4f, A=%5.3f ' % tuple(popt_Kalapus[0]))
        


    except RuntimeError:
        r2_LG_list.append(0)
        pass



def exp_fit(x, y):
    
    x_data = np.asarray(x)
    y_data = np.asarray(y)
    
    y_final = max(y)
    y_0 = y[0]
    diff_yf_y0 = abs(y_final-y_0)
    
    

    def exp_eq(x, k):
        if y_final > 0:
            y = (y_final*(1-np.exp(-k*x)))

        
        return y
    

    
    
    if y_final > 0:
        bounds_range = (0, [5])

        
    try:
        
        
        popt, pcov = curve_fit(exp_eq, x_data, y_data, bounds=bounds_range)
        y_exp_list_test.append(exp_eq(x_data, *popt))
        

        
        r2_exp = r2_score(y_data, exp_eq(x_data, *popt))
        exp_list.append(r2_exp)
        popt_exp.append(popt)
        y_exp_list = []
        y_exp_list.append(exp_eq(x_data, *popt))

        
        
        print('Pseudo first order R2: ', round(r2_exp, 3))
        print('Pseudo first order k: ', popt_exp[0][0])

            
    except RuntimeError:
        exp_list.append(0)
        pass
           

n = 0
w = 0


while z < (len(df.index)+1):

    starts1 += add_number
    starts2 += add_number
    z += add_number
    print(starts1, starts2)


    for i in range(2, len(df.columns)):
        y = df.iloc[starts1:starts2, i]
        x = df.iloc[starts1:starts2, 1]
        all_y.append(y)
        tit = (df.iloc[starts1][0], w)
        w += 1
        x = x.tolist()
        y = y.tolist()


        logaritmic = np.polyfit(np.log(x), y, 1)
        a = round(logaritmic[0],3)
        b= round(logaritmic[1],3)
        
        coeff_log.append(logaritmic)

        y_log_list = []

        for _ in x:
            y_log = coeff_log[0][0]*math.log(_)+coeff_log[0][1]
            y_log_list.append(y_log)


        r2_log_scipy = r2_score(y, y_log_list)

        print('Logaritmic R2: ', round(r2_log_scipy, 3))
        print(f'Logarimitc coeff (y = a*log(x) + b): a={a}, b={b}')


        LG_fit(x, y)
        exp_fit(x, y)




y_exp_list = y_exp_list_test
y_kal1 = y_kal[0]
y_kal1 = y_kal1.tolist()
y_exp_list1 = y_exp_list[0]
y_exp = y_exp_list1.tolist()

plt.figure(figsize=(4.33, 4.33))

df = pd.DataFrame(list(zip(x, y, y_log_list, y_kal1)))
kwargs = {'linewidth':4}




plt.plot(x, y_kal1, "-r", label='Kalapus equation', linewidth=1.5) 
plt.plot(x, y_exp, label='pseudo first-order equation', linewidth=1.5, color='#008000')
plt.plot(x, y_log_list, '-', color='#A0522D', label='logarithmic equation', linewidth=1.5)
plt.plot(x, y, "ob", label='experimental data', markersize=5, linewidth=1.5)

plt.legend(loc='best', prop={'size':7})
plt.xlabel('time [h]', fontsize=8)
plt.ylabel('% of dissolved solid', fontsize=8)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.title('ZnO', loc='center', fontsize=10)
plt.title(f'{sheet_name}', loc='center', fontsize=15)



'FILE WILL BE SAVE IN YOUR WORKING DIRECTORY AS PNG'
saving = int(input("File will be saved in your working directory, in the same folder in which you run this script. Want to save plot?  Yes = 0, no = 1: "))

if saving == 0:    
    plt.savefig(f'{sheet_name}.png', dpi=300) 



y_final = max(y)
n = 2

pred_or_not = int(input('Here u can spcify if you want to predict solubility in future: yes = 0, no = 1: '))
if pred_or_not == 0:
    
    time_one = float(input('Specify first time at which you want to predict solubility: '))
    time_two = float(input('Specify second time at which you want to predict solubility: '))
    xpr = [time_one, time_two]
    
    
    
    
    def pred_func(A, k, B, y_final, x):
    
    
        y_pred_list = []
        for i in range(len(xpr)):
    
            y_pred = (y_final*(B-np.exp(-k*xpr[i])))*A
            y_pred_list.append(y_pred)
    
        return y_pred_list
    
    
    A = popt_Kalapus[0][0]
    k = popt_Kalapus[0][1]
    B = popt_Kalapus[0][2]



    y_pred_kalapus = pred_func(A, k, B, y_final, xpr)
    print(y_pred_kalapus)

