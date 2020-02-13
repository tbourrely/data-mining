#!/usr/bin/env python
# -*- coding: utf-8 -*-

from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# convert categorical values into one-hot vectors and ignore ? values
# corresponding to missing values
supermarket_one_hot = pd.get_dummies(supermarket)
supermarket_one_hot.drop(supermarket_one_hot.filter(regex='_\?$',axis=1).columns,axis=1,inplace=True)

# option to show all itemsets
pd.set_option('display.max_colwidth', -1)



# select rules with more than 2 antecedents
# rules.loc[map(lambda x: len(x)>2,rules['antecedents'])]


