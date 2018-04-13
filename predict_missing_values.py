import numpy as np
import pandas as pd

def predict_missing_values(df, moving_average_window = 2, moving_average_weight = 1, polynomial_weight = 1.5):
    def predict_column(col):
        col = col.copy(deep=True)
        miss = col[col > col.mean()]
        while col[col > col.mean()].size > 0:
            indexes = miss.dropna().index
            min_index = np.argmax(np.diff(indexes.values))
            i1 = indexes[min_index]+1
            i2 = indexes[min_index + 1]-1
            
            subset = col.iloc[i1:i2]

            p_pred_i1 = polynomial_predict(subset)
            m_pred_i1 = moving_average_predict(subset, moving_average_window)
            final_pred_i1 = np.average([p_pred_i1, m_pred_i1], weights=[polynomial_weight, moving_average_weight])
            
            col.loc[i1] = final_pred_i1
            
            
            subset = subset[::-1]
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
    
    predictions = []
    for i in range(1, int(roots/2)):
        power = range(roots-i,roots+i)
        p = [np.poly1d(np.polyfit(range(data.values.size), data.values, i))(data.shape[0]) for i in power]
        predictions.append(np.mean(p))

    return np.mean(predictions)

def moving_average_predict(data, window=2):
    smooth = data.rolling(window=window).mean()[window:]
    return np.mean(smooth.iloc[-window])