__author__ = 'Joshua'
import xlrd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import numpy as np




def get_data_array(city_data):
    idx = 0
    for i in range(2,city_data.nrows):
        per_set = []
        # total_set.insert(0,city_data.col_values(1)[i])
        area_val = city_data.col_values(2)[i]
        price_val = city_data.col_values(3)[i]
        revenue_val = city_data.col_values(4)[i]

        if(area_val == '--'):
            total_area.insert(0, 0)
        else:
            total_area.insert(0, area_val * 10e3)

        if(price_val == '--'):
            unit_price.insert(0, 0)
        else:
            unit_price.insert(0, price_val)

        if(revenue_val == '--'):
            total_revenue.insert(0, 0)
        else:
            total_revenue.insert(0, revenue_val * 10e7)


        total_idx.append(idx)
        # per_set.append(city_data.col_values(1)[i])
        per_set.append(city_data.col_values(2)[i])
        per_set.append(city_data.col_values(3)[i])
        totalX.insert(0, per_set)
        idx = idx + 1


def one_vs_one_model():

    reg2 = linear_model.LinearRegression()
    reg3 = linear_model.LinearRegression()


    reg2.fit(total_idx.reshape(-1,1), total_area.reshape(-1,1))
    reg3.fit(total_idx.reshape(-1,1), unit_price.reshape(-1,1))

    prediction_linear(reg2, reg3)

def prediction_linear(reg2, reg3):
    test1 = []
    test2 = []

    plot_all_predict_area = []
    plot_all_predict_price = []
    plot_all_predict_monthly_rev = []
    for i in range(total_area.size+1, total_area.size+13):
        test1.append(i)

    for f in range(0, total_area.size+13):
        test2.append(f)

    tmp_total = 0
    print '\n'
    print 'Linear Prediction'
    for j in range(0, len(test1)):

        predict_area = reg2.predict(test1[j])
        predict_price = reg3.predict(test1[j])
        monthly_revenue = predict_area*predict_price
        tmp_total+= monthly_revenue
        print 'Predicted Total Area: ', predict_area, ' Predicted Unit Price: ', predict_price, ' Monthly Revenue: ', monthly_revenue
    print 'Whole Year Total Revenue: ', tmp_total

    print '\n'

    for p in range(0, len(test2)):

        all_predict_area = reg2.predict(test2[p])
        all_predict_price = reg3.predict(test2[p])
        all_monthly_revenue = all_predict_area * all_predict_price

        plot_all_predict_area.append(all_predict_area[0])
        plot_all_predict_price.append(all_predict_price[0])
        plot_all_predict_monthly_rev.append(all_monthly_revenue[0])


    fig= plt.figure(figsize=(12,9))
    # fig.suptitle(data.sheet_names())
    # plt.subplot(4,1,1)
    # plt.title('total units')
    # plt.plot(total_set)
    plt.subplot(3, 1, 1)
    plt.title('total area')
    plt.plot(total_area)
    plt.plot(plot_all_predict_area, 'r')
    plt.subplot(3, 1, 2)
    plt.title('unit price')
    plt.plot(unit_price)
    plt.plot(plot_all_predict_price, 'r')
    plt.subplot(3, 1, 3)
    plt.title('total revenue')
    plt.plot(total_revenue)
    plt.plot(plot_all_predict_monthly_rev, 'r')
    # plt.savefig(savePath+data.sheet_names()+'_linear.png')
    plt.show()


def one_vs_one_poly():

    poly1 = Pipeline([('poly', PolynomialFeatures(degree=3)), ('linear', LinearRegression(fit_intercept=False))])
    poly1 = poly1.fit(total_idx.reshape(-1,1), total_area.reshape(-1,1))
    coeff1 = poly1.named_steps['linear'].coef_

    poly2 = Pipeline([('poly', PolynomialFeatures(degree=5)), ('linear', LinearRegression(fit_intercept=False))])
    poly2 = poly2.fit(total_idx.reshape(-1, 1), unit_price.reshape(-1, 1))
    coeff2 = poly2.named_steps['linear'].coef_

    prediction_poly(poly1, poly2)


def prediction_poly(poly1, poly2):
    test1 = []

    test2 = []
    plot_all_predict_area = []
    plot_all_predict_price = []
    plot_all_predict_monthly_rev = []

    for i in range(total_area.size+1, total_area.size+13):
        test1.append(i)

    for f in range(0, total_area.size+13):
        test2.append(f)

    tmp_total = 0

    print'\n'
    print'Poly Prediction'
    for j in range(0, len(test1)):

        predict_area = poly1.predict(test1[j])
        predict_price = poly2.predict(test1[j])
        monthly_revenue = predict_area * predict_price
        tmp_total += monthly_revenue
        print 'Predicted Total Area: ', predict_area, ' Predicted Unit Price: ', predict_price, ' Monthly Revenue: ', monthly_revenue
    print 'Whole Year Total Revenue: ', tmp_total
    print '\n'

    for p in range(0, len(test2)):

        all_predict_area = poly1.predict(test2[p])
        all_predict_price = poly2.predict(test2[p])
        all_monthly_revenue = all_predict_area * all_predict_price


        plot_all_predict_area.append(all_predict_area[0])
        plot_all_predict_price.append(all_predict_price[0])
        plot_all_predict_monthly_rev.append(all_monthly_revenue[0])



    fig =plt.figure(figsize=(12,9))
    fig.suptitle("Nanjing", fontweight='bold')
    # plt.subplot(4,1,1)
    # plt.title('total units')
    # plt.plot(total_set)
    plt.subplot(3, 1, 1)
    plt.title('total area')
    plt.plot(total_area)
    plt.plot(plot_all_predict_area, 'r')
    plt.subplot(3, 1, 2)
    plt.title('unit price')
    plt.plot(unit_price)
    plt.plot(plot_all_predict_price, 'r')
    plt.subplot(3, 1, 3)
    plt.title('total revenue')
    plt.plot(total_revenue)
    plt.plot(plot_all_predict_monthly_rev, 'r')
    # plt.savefig(savePath + data.sheet_names() + '_poly.png')
    plt.show()



file = "raw_data_english.xlsx"
savePath = "images/"
data = xlrd.open_workbook(file)
log = open(savePath+"Results.txt", "w")

data_nanjing = data.sheet_by_index(11)
data_nanning = data.sheet_by_index(12)
data_qingdao = data.sheet_by_index(14)
data_sanya = data.sheet_by_index(15)

total_set = []
total_area = []
unit_price = []
total_revenue = []
totalX = []
total_idx = []



get_data_array(data_qingdao)

total_idx = np.asarray(total_idx)
total_set = np.asarray(total_set)
total_area = np.asarray(total_area)
total_revenue = np.asarray(total_revenue)
unit_price = np.asarray(unit_price)

print "City Name: Qingdao"

one_vs_one_model()
one_vs_one_poly()
