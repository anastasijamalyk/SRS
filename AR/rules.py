import pandas as pd
import numpy as np
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth,association_rules
import itertools


def mine_rules(seq, min_support=0.000013,  min_threshold=0.01):    #
        te = TransactionEncoder()
        te_ary = te.fit([list(i) for i in seq]).transform([list(i) for i in seq])
        # set_trace()
        df = pd.DataFrame(te_ary, columns=te.columns_)
        frequent_itemsets = fpgrowth(df, min_support=min_support, use_colnames=True)     #mining frequent itemsets
        as_rul=association_rules(frequent_itemsets, metric="confidence", min_threshold=min_threshold)  #generating association rules
        rules=as_rul[(as_rul['antecedents'].map(len) == 1) & (as_rul['consequents'].map(len) == 1)]                                     #leaving rules with 1 antecedent and 1 consequent (pairs)
        rules['antecedents']=[list(i) for i in rules['antecedents']]                          #converting frozensets to sets
        rules['consequents']= [list(i) for i in rules['consequents']]                         #converting frozensets to sets
        rules['seq']=rules['antecedents']+rules['consequents']                                #adding antecedents+consequents
        rules_fin=frozenset(tuple(i) for i in rules['seq'])      #final rules in the form [sequence, support]
        print("Total number of rules mined: ", len(rules_fin))
        return rules_fin



def freq_item(seq, rules):
        perm=[]
        for i in seq:
          perm.append(list(itertools.permutations(i,2))) #generates all possible permutations of size=2 for each sequence
        res=[]
        for sets in perm:
            res.append([x for x in sets if x in rules])   # appends only those pairs that are present in rules
        for i in range(len(res)):
            if not res[i]:
              res[i]=perm[i]                            #if sequence has no frequent itempairs, we leave it as is
        result=[]
        for id, row in enumerate(res):
            result.append([i for sub in res[id] for i in sub])  #unfold tuples into a list of items   
        padded=zip(*itertools.zip_longest(*result,fillvalue=0)) #padding to get a fixed sequence length
        fin_seq=np.array(list(padded))
        return fin_seq