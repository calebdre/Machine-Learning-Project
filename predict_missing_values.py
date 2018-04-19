import numpy as np
import pandas as pd
from IPython.core.debugger import set_trace
from multiprocessing import Pool

def normalize(val, mmin, mmax):
    return (val - mmin) / (mmax - mmin)

def denormalize(normalized, mmin, mmax):
    return normalized * (mmax - mmin) + mmin

def predict_missing_values(df, moving_average_window = 2, moving_average_weight = 1, polynomial_weight = 1.5):
    def predict_column(col):
        print("***************\n{}\n***************\n".format(col.name))
        col = col.copy(deep=True).dropna()
        miss = col[col > col.mean()]
        m = col.mean()
        while col[col > m].size > 0:
            indexes = col[col > col.mean()].dropna().index
            single= False
            #print(indexes.size)
            if col[col > col.mean()].dropna().size == 1:
                i1 = -1
                i2 = indexes[0]
                single = True
            else:
                min_index = np.argmax(np.diff(indexes.values))
                i1 = indexes[min_index]
                i2 = indexes[min_index + 1]
                
            subset = col.iloc[i1+1:i2]
            if subset.size == 0:
                i2 = i1
                i1 -= 15
                subset = col.iloc[i1:i2]
                single = True
            
            if subset.size < 5:
                i1 = indexes.min()
                subset = col.iloc[0: i1]
            
            try:
                p_pred_i1, rc = polynomial_predict(subset)
                m_pred_i1 = moving_average_predict(subset, moving_average_window)
                final_pred_i1 = moving_average_predict(subset, moving_average_window)
                final_pred_i1 = np.average([p_pred_i1, m_pred_i1], weights=[polynomial_weight, moving_average_weight])
                
            except Exception as e:
                final_pred_i1 = moving_average_predict(subset, moving_average_window)
            
            if single:
                col.loc[i2] = final_pred_i1
            else:
                col.loc[i1] = final_pred_i1
            
            subset = subset[::-1].dropna()
            
            p_pred_i2 = polynomial_predict(subset)
            m_pred_i2 = moving_average_predict(subset, moving_average_window)
            final_pred_i2 = np.average([p_pred_i2, m_pred_i2], weights=[polynomial_weight, moving_average_weight])
            
            col.loc[i2] = final_pred_i2
        
        return col
    
    h = df.apply(predict_column)
    print("***********************\n\nDone!\n\n***********************\n")
    return h


def polynomial_predict(data):
    # checks every pair and compares the each point to see if the mean is between the two. If it is, we've found a root
    roots = data.rolling(window=2).apply(lambda x: (x[0] > data.mean() and x[1] < data.mean()) or (x[0] < data.mean() and x[1] > data.mean())).dropna()
    roots = roots[roots > 0].size
    root_changed = False
    if roots > 10:
        roots = 10
        root_changed = True
        
    mmin =  data.min() / 10
    mmax =  data.max() / 10
    
    data = normalize(data, mmin, mmax)
    coeffs = np.polyfit(range(data.values.size), data.values, roots)
    f = np.poly1d(coeffs)
    
    pred = f(data.shape[0])
    return denormalize(pred, mmin, mmax), root_changed
    #return pred, root_changed
    
def moving_average_predict(data, window=2):
    smooth = data.rolling(window=window).mean()[window:]
    return np.mean(smooth.iloc[-window])