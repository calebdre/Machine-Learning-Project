from math import sqrt, floor
import numpy as np
import pandas as pd

class RandomForest:
    def __init__(self):
        self.trees = []
    
    def train(self, origin_df, labels, num_trees=10, num_features=None, num_sample_rows=None , max_tree_depth=20, min_split_samples=5):
        if num_features is None or num_features < 1:
            num_features = floor(sqrt(origin_df.shape[1]))
            
        if num_sample_rows is None or num_sample_rows < 1:
            num_sample_rows = floor(origin_df.shape[0] / 4)

        for i in range(num_trees):
            features = np.random.choice(origin_df.columns, size=num_features, replace=False)
            rows = np.random.choice(origin_df.shape[0], size=num_sample_rows, replace=False)

            df = origin_df[features]
            df = df.iloc[rows]

            print("\n*** Creating tree #{} ***\n".format(i+1))
            self.trees.append(DecisionTree().train(df, labels, max_tree_depth, min_split_samples))

        return self
    
    def predict(self, row):
        if not isinstance(row, pd.Series):
            raise Exception("`row` must be an instance of Pandas.Series")
        
        predictions = [tree.predict(row) for tree in self.trees]
        return np.bincount(predictions).argmax()

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
    def __init__(self):
        self.root = None
    
    def train(self, origin_df, labels, max_depth = 20, min_split_samples = 10):
        def classify_branch(df, labels):
            return labels.merge(df, left_index=True, right_index=True).groupby("label").count().sum(axis=1).idxmax()
        
        def train_recurse(df, labels, depth):
            print("parsing @ depth {}".format(depth))
            col, val = self.calc_best_gini_split(df, labels)
            left_split = df[df[col] > val]
            right_split = df[df[col] < val]

            if depth > max_depth or np.minimum(left_split.shape[0], right_split.shape[0]) < min_split_samples:
                classification = classify_branch(df, labels)
                print("classified {} at depth {} to be in class {}".format(col, depth, classification))
                return classification

            node = Node(col, val)
            node.left = train_recurse(left_split, labels, depth + 1)
            node.right = train_recurse(right_split, labels, depth + 1)

            print("Done with depth {}".format(depth))
            return node
        
        self.root = train_recurse(origin_df, labels, 1)
        return self
    
    def calc_best_gini_split(self, df, labels):
        total_rows = df.shape[0]

        def calc(col_df):
            min_cost = 2
            min_cost_index = -1
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
            return pd.Series({"cost": min_cost, "index": min_cost_index})
        
        splits = df.apply(calc, axis=0)
        best_split_col = splits.T["cost"].idxmin()
        best_split_index = splits[best_split_col]["index"]

        return best_split_col, df[best_split_col][best_split_index]
    
    def predict(self, row):
        if self.root is None: 
            raise Exception("Must call `train` before using `predict`")
        
        node = self.root
        while type(node) != np.int64:
            node = node.traverse(row)
        
        return node

