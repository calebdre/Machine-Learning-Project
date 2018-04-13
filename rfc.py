from math import sqrt, floor
import numpy as np
import pandas as pd
from multiprocessing import Pool
from IPython.core.debugger import set_trace

def calc(data):
    col_df, labels = data
    min_cost = 2
    min_cost_index = -1
    total_rows = col_df.shape[0]
    for index, row in col_df.iteritems():
        split1 = col_df[row > col_df]
        split2 = col_df[row < col_df]

        s1_grouped = labels.merge(pd.DataFrame(split1), left_index=True, right_index=True).groupby("label")
        s2_grouped = labels.merge(pd.DataFrame(split2), left_index=True, right_index=True).groupby("label")

        s1_group_count = s1_grouped.count()
        s2_group_count = s2_grouped.count()

        s1_cost = (1 - np.power(s1_group_count / s1_group_count.sum(), 2).sum(axis=0)) * (split1.shape[0] / total_rows)
        s2_cost = (1 - np.power(s2_group_count / s2_group_count.sum(), 2).sum(axis=0)) * (split2.shape[0] / total_rows)

        total_cost = (s1_cost + s2_cost).iloc[0]
        if total_cost < min_cost:
            min_cost = total_cost
            min_cost_index = index
    return (min_cost, min_cost_index, col_df.name)

def calc_best_gini_split(df, labels):
    total_rows = df.shape[0]
    
    columns = [(df.iloc[:, i], labels) for i in range(df.shape[1])]
    with Pool(4) as pool:
        result = pool.map_async(calc, columns)
        mins = np.array(result.get())
        
        mins_best_index = mins[:, 0].argmin()
        best_split = mins[mins_best_index, :]
        best_split_col = best_split[2]
        best_split_index = best_split[1]
        
        return best_split_col, df[best_split_col][int(best_split_index)]

class RandomForest:
    def __init__(self, verbose=False):
        self.trees = []
        self.verbose = verbose
    
    def train(self, origin_df, labels, num_trees=10, num_features=None, num_sample_rows=None , max_tree_depth=20, min_split_samples=5, bias_class=None, bias_amount=0):     
        if num_features is None or num_features < 1:
            num_features = floor(sqrt(origin_df.shape[1]))
            
        if num_sample_rows == None or num_sample_rows < 1:
            num_sample_rows = floor(origin_df.shape[0] / 4)
        
        tree_selections = []
        if bias_class != None and bias_amount > 0:
            # create special trees for the classes that aren't in bias
            # designate how many trees will be allocated = n
            num_special_trees = floor(bias_amount * num_trees)
            # select subset of rows without bias class
            special_tree_labels = labels[labels != bias_class].dropna()
            special_tree_rows = origin_df.iloc[special_tree_labels.index]
            # train on those
            self.p("************", "Bias class found - removing {} class and building {} of {} trees".format(bias_class, num_special_trees, num_trees),"************")
            tree_selections += self.train(special_tree_rows, special_tree_labels, num_special_trees, num_features, num_special_trees, max_tree_depth, min_split_samples)
            num_trees -= num_special_trees
            self.p("***********", "Finished creating bias trees. Creating {} more trees from all the data".format(num_trees), "************")

        for i in range(num_trees):
            total_labels = labels["label"].unique().size
            rows = np.random.choice(origin_df.shape[0], size=num_sample_rows, replace=False)
            
            while labels.iloc[rows]["label"].unique().size != total_labels:
                rows = np.random.choice(origin_df.shape[0], size=num_sample_rows, replace=False)
                    
            features = np.random.choice(origin_df.columns, size=num_features, replace=False)
            
            df = origin_df[features]
            df = df.iloc[rows]
            
            tree_selections.append(df)
            
            self.p("*** Creating tree #{} ***".format(i+1))
            self.trees.append(DecisionTree(self.verbose).train(df, labels, max_tree_depth, min_split_samples))
            
        return tree_selections
    
    def predict(self, row):
        if not isinstance(row, pd.Series):
            raise Exception("`row` must be an instance of Pandas.Series")
        
        
        predictions = [tree.predict(row) for tree in self.trees]
        return np.bincount(predictions).argmax()
            
    def p(self, *args):
        if self.verbose:
            for arg in args:
                print(" ")
                print(arg)
                print(" ")
        

class Node:
    def __init__(self, column, value):
        self.col = column
        self.val = value
        self.left = None
        self.right = None
    
    def traverse(self, row):
        if row[self.col] > self.val:
            return self.left
        else:
            return self.right

class DecisionTree:
    def __init__(self, verbose=False):
        self.root = None
        self.verbose = verbose
    
    def train(self, origin_df, labels, max_depth = 20, min_split_samples = 10):
        def classify_branch(df, labels):
            return labels.merge(df, left_index=True, right_index=True).groupby("label").count().sum(axis=1).idxmax()
        
        def train_recurse(df, labels, depth):
            self.p("parsing @ depth {}".format(depth))
            col, val = calc_best_gini_split(df, labels)
            left_split = df[df[col] > val]
            right_split = df[df[col] < val]

            if depth > max_depth or np.minimum(left_split.shape[0], right_split.shape[0]) < min_split_samples:
                classification = classify_branch(df, labels)
                self.p("classified {} at depth {} to be in class {}".format(col, depth, classification))
                return classification

            node = Node(col, val)
            node.left = train_recurse(left_split, labels, depth + 1)
            node.right = train_recurse(right_split, labels, depth + 1)

            self.p("Done with depth {}".format(depth))
            return node
        
        self.root = train_recurse(origin_df, labels, 1)
        return self

    def p(self, *args, important=False):
        if self.verbose:
            for arg in args:
                print(" ")
                print(arg)
                print(" ")

    def predict(self, row):
        if self.root is None: 
            raise Exception("Must call `train` before using `predict`")
        
        node = self.root
        while not np.isscalar(node):
            node = node.traverse(row)
        
        return node

