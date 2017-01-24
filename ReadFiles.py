from __future__ import print_function
import xlrd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import numpy as np


file = "raw_data_english.xlsx"
data = xlrd.open_workbook(file)

sheet=data.sheet_names()[0]
savePath = "images/"
zhengzhou = data.sheet_by_index(3)
log = open(savePath+"Results.txt", "w")





for i in range(0, data.nsheets):
    print (""+data.sheet_names()[i], file = log)
    # print '/'+ data.sheet_names()[i]+'_linear.png'