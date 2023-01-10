import pandas as pd
import numpy as np
import operator
from scipy import stats
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.linear_model import LinearRegression
import math

def comparisonStats(model_data, exp_data1, exp_data2, header, values=None, operators=None, display=False, correct_pred=-1, headers=-1, percentile=0):
    '''
    comparisonStats: Gives all stats available to compare two explanation sets (pandas dataframe)
    model_data: model training/testing data (pandas dataframe)
    exp_data: explanation results (pandas dataframe)
    header: column of explanations to compare (string)
    values: used with operators to filter stats (array)
    operators: used with values to fiter stats (array of string operators) [==, <=, >=, >, <, !=]
    display: display comparison as graphs (boolean)
    correct_pred: filter to only correct/incorrect prediction 
        (default: not filtering, True: filter only correct predictions, False: filter only incorrect predictions) 
    headers: used with values and operators to filter data on specific headers (array of strings)
    percentile: used for removing outliers (float)
    '''
    filtered_data = model_data
    if header not in (model_data.columns):
        print(header, 'is not in the headers')
        return
    if not (operators == None and values == None):
        ops = {'==':operator.eq, '<=':operator.le, '>=':operator.ge, '!=':operator.ne, '>':operator.gt, '<':operator.lt}
        if type(operators) != list:
            operators = [operators]
        if type(values) != list:
            values = [values]
        for oper in operators:
            if oper not in ops.keys():
                print(operator, 'is not a valid operator')
                return
        if headers == -1:
            headers = [header] * len(values)
        if len(values)!=len(operators):
            print('number of values and operators are not equal')
            return

        for oper, value, h in zip(operators, values, headers):
            filtered_data = filtered_data[ops[oper](filtered_data[h], value)]
    if correct_pred == True:
        filtered_data = filtered_data[filtered_data['prediction'] == filtered_data['income']]
    if correct_pred == False:
        filtered_data = filtered_data[filtered_data['prediction'] != filtered_data['income']]
    
    indecies = filtered_data['index']
    filtered_exp1 = exp_data1[exp_data1['index'].isin(indecies)]
    f_exp1 = filtered_exp1[header].values
    filtered_exp2 = exp_data2[exp_data2['index'].isin(indecies)]
    f_exp2 = filtered_exp2[header].values
    exp_dif = f_exp1-f_exp2

    sample_size = len(f_exp1)
    bins = max(min(sample_size,150),10)
    dif_outliers = exp_dif[np.percentile(exp_dif,percentile)<=exp_dif]
    dif_outliers = dif_outliers[dif_outliers<=np.percentile(exp_dif,100-percentile)]
    n,plt_bins,_ = plt.hist(dif_outliers,bins=bins)

    dif_results = {}
    dif_results['mean'] = np.mean(exp_dif)
    dif_results['std'] = np.std(exp_dif)
    dif_results['variance'] = np.var(exp_dif)
    # dif_results['z-score'] = np.mean(stats.zscore(exp_dif))
    dif_results['median'] = np.median(exp_dif)
    dif_results['mode*'] = (plt_bins[n.argmax()]+plt_bins[n.argmax()+1])/2
    dif_results['importance'] = np.mean(np.abs(exp_dif))

    exp1_outliers = f_exp1[np.percentile(f_exp1,percentile)<=f_exp1]
    exp1_outliers = exp1_outliers[exp1_outliers<=np.percentile(f_exp1,100-percentile)]
    bin_index1,plt_bins1,_ = plt.hist(exp1_outliers,bins=bins)
    results1 = {}
    results1['mean'] = np.mean(f_exp1)
    results1['std'] = np.std(f_exp1)
    results1['variance'] = np.var(f_exp1)
    # results1['z-score'] = np.mean(stats.zscore(f_exp1))
    results1['median'] = np.median(f_exp1)
    results1['mode*'] = (plt_bins1[bin_index1.argmax()]+plt_bins1[bin_index1.argmax()+1])/2
    results1['importance'] = np.mean(np.abs(f_exp1))

    exp2_outliers = f_exp2[np.percentile(f_exp2,percentile)<=f_exp2]
    exp2_outliers = exp2_outliers[exp2_outliers<=np.percentile(f_exp2,100-percentile)]
    bin_index2,plt_bins2,_ = plt.hist(exp2_outliers,bins=bins)
    results2 = {}
    results2['mean'] = np.mean(f_exp2)
    results2['std'] = np.std(f_exp2)
    results2['variance'] = np.var(f_exp2)
    # results2['z-score'] = np.mean(stats.zscore(f_exp2))
    results2['median'] = np.median(f_exp2)
    results2['mode*'] = (plt_bins2[bin_index2.argmax()]+plt_bins2[bin_index2.argmax()+1])/2
    results2['importance'] = np.mean(np.abs(f_exp2))

    comp_results = {}
    comp_results['sign_match'] = np.mean((f_exp1>=0)==(f_exp2>=0)) # Both are the same
    comp_results['sign_pos'] = np.mean(np.all(np.stack(( ((f_exp1>=0)==1) , ((f_exp2>=0)==0) ), axis=0), axis=0)) # 1 is pos and 2 is neg
    comp_results['sign_neg'] = np.mean(np.all(np.stack(( ((f_exp1>=0)==0) , ((f_exp2>=0)==1) ), axis=0), axis=0)) # 1 is neg and 2 is pos
    comp_results['importance'] = np.mean(np.abs(f_exp1)) - np.mean(np.abs(f_exp2))
    comp_results['importance_sign_match'] = np.mean(((np.abs(f_exp1)-np.abs(f_exp2))[((f_exp1>=0)==(f_exp2>=0))])>0) # When Signs Match is A always more important
    comp_results['correlation'] = np.corrcoef(f_exp1, f_exp2)[0,1] # When A increases does B increase
    # comp_results['one-to-one'] = # measure of variance given a value
    bin_index1,plt_bins1,_ = plt.hist(exp1_outliers,bins=40)
    bin_hist1 = []
    for iter, i in enumerate(bin_index1.astype(int)):
        if iter<len(plt_bins1):
            bin_hist1.extend([plt_bins1[iter]]*i)
    trim_to = min(len(bin_hist1), len(exp2_outliers))
    bin_vals = defaultdict(list)
    for bin, y in zip(bin_hist1[:trim_to], exp2_outliers[:trim_to]):
        bin_vals[bin].append(y)
    total = []
    for values in bin_vals.values():
        std = stats.tstd(values)
        if len(values)>1:
            total.extend([std]*len(values))
    comp_results['inverse-correlation'] = np.mean(total) # Std of B within a bin of A

    linearModel = LinearRegression()
    linearModel.fit(exp1_outliers.reshape(-1,1), exp2_outliers)
    comp_results['slope'] = (linearModel.coef_)[0]
    p1 = np.array([0,linearModel.intercept_])
    p2 = np.array([1,linearModel.intercept_+comp_results['slope']])
    linear_distance = 0
    for p3 in np.stack((exp1_outliers, exp2_outliers), axis = 1):
        linear_distance += abs(np.cross(p2-p1,p3-p1)/np.linalg.norm(p2-p1))
    comp_results['linear_distance'] = linear_distance/len(exp1_outliers)
    comp_results['sine_distance'] = abs(math.sin(math.atan((1-comp_results['slope'])/(1+comp_results['slope']))))
    mid_point = 1
    steepness = 3
    comp_results['normalized_distance'] = (1/(1+math.pow(math.e, -steepness*(comp_results['linear_distance']-mid_point))))
    comp_results['linear_disagreement'] = (comp_results['sine_distance']+(1/(1+math.pow(math.e, -steepness*(comp_results['linear_distance']-mid_point)))))/2


    results = {}
    results['difference'] = dif_results
    results[exp_data1['model'].values[0]] = results1
    results[exp_data2['model'].values[0]] = results2
    results['comparative'] = comp_results

    plt.close()
    if (display == True):
        # plt.hist(dif_outliers,bins=bins)
        plt.hist(exp1_outliers,bins=bins)
        plt.hist(exp2_outliers,bins=bins)
        plt.legend(['Difference',exp_data1['model'].values[0],exp_data2['model'].values[0]])
        plt.show()
        trim_to = min(len(exp1_outliers),len(exp2_outliers))
        plt.scatter(exp1_outliers[:trim_to], exp2_outliers[:trim_to], s=1)
        plt.xlabel(exp_data1['model'].values[0])
        plt.ylabel(exp_data2['model'].values[0])
        plt.show()
        mean_exp = (exp1_outliers+exp2_outliers)/2
        dif_exp = exp1_outliers-exp2_outliers
        plt.scatter(mean_exp[:trim_to], dif_exp[:trim_to], s=1)
        plt.xlabel('mean')
        plt.ylabel('difference')
        plt.show()
        
        
    return results 



def agreementScore(model_data, exp_data1, exp_data2, avoid_headers=[], percentile=0):
    '''
    comparisonStats: Gives all stats available to compare two explanation sets (pandas dataframe)
    model_data: model training/testing data (pandas dataframe)
    exp_data: explanation results (pandas dataframe)
    avoid_headers: list of headers to skip (array of strings)
    percentile: used for removing outliers (float)
    '''
    filtered_data = model_data
    header_total = 0
    score_sum = 0
    
    for header in exp_data1.columns:
        if header in avoid_headers:
            continue
        indecies = filtered_data['index']
        filtered_exp1 = exp_data1[exp_data1['index'].isin(indecies)]
        f_exp1 = filtered_exp1[header].values
        filtered_exp2 = exp_data2[exp_data2['index'].isin(indecies)]
        f_exp2 = filtered_exp2[header].values
        if len(f_exp1) != len(f_exp2):
            continue

        exp1_outliers = f_exp1[np.percentile(f_exp1,percentile)<=f_exp1]
        exp1_outliers = exp1_outliers[exp1_outliers<=np.percentile(f_exp1,100-percentile)]
        exp2_outliers = f_exp2[np.percentile(f_exp2,percentile)<=f_exp2]
        exp2_outliers = exp2_outliers[exp2_outliers<=np.percentile(f_exp2,100-percentile)]
        comp_results = {}

        linearModel = LinearRegression()
        linearModel.fit(exp1_outliers.reshape(-1,1), exp2_outliers)
        comp_results['slope'] = (linearModel.coef_)[0]
        p1 = np.array([0,linearModel.intercept_])
        p2 = np.array([1,linearModel.intercept_+comp_results['slope']])
        linear_distance = 0
        for p3 in np.stack((exp1_outliers, exp2_outliers), axis = 1):
            linear_distance += abs(np.cross(p2-p1,p3-p1)/np.linalg.norm(p2-p1))
        comp_results['linear_distance'] = linear_distance/len(exp1_outliers)
        comp_results['sine_distance'] = abs(math.sin(math.atan((1-comp_results['slope'])/(1+comp_results['slope']))))
        mid_point = 1
        steepness = 3
        comp_results['normalized_distance'] = (1/(1+math.pow(math.e, -steepness*(comp_results['linear_distance']-mid_point))))
        comp_results['linear_disagreement'] = (comp_results['sine_distance']+comp_results['normalized_distance'])/2
        header_total += 1
        score_sum += comp_results['linear_disagreement']
    return score_sum/header_total
        
