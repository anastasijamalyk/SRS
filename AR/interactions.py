
import numpy as np
import scipy.sparse as sp
import collections
from rules import *

class Interactions(object):
    """
    Interactions object. Contains (at a minimum) pair of user-item
    interactions. This is designed only for implicit feedback scenarios.
    Parameters
    ----------
    file_path: file contains (user,item,rating) triplets
    user_map: dict of user mapping
    item_map: dict of item mapping
    """
    
    def __init__(self, file_path,
                 min_support,
                 min_threshold,
                 user_map=None,
                 item_map=None):
        test_flag = False
        if not user_map and not item_map:
            user_map = dict()
            item_map = dict()

            num_user = 0
            num_item = 0
        else:
            num_user = len(user_map)
            num_item = len(item_map)
            test_flag = True
        user_ids = list()
        item_ids = list()
        # read users and items from file

        with open(file_path, 'r') as fin:
            for line in fin:
                u, i, _ = line.strip().split()
                user_ids.append(u)
                item_ids.append(i)

        # update user and item mapping
        for u in user_ids:
            if u not in user_map:
                user_map[u] = num_user   # adding values to user_map dict, keys are ids stripped from doc
                num_user += 1
        for i in item_ids:
            if i not in item_map:
                item_map[i] = num_item  #adding values to item_map dict
                num_item += 1

        user_ids = np.array([user_map[u] for u in user_ids]) 
        #print(user_ids)    [   0    0    0 ... 6039 6039 6039]
        item_ids = np.array([item_map[i] for i in item_ids])
        #print(item_ids)     [   0    1    2 ... 1450   87 2747]
        self.num_users = num_user
        self.num_items = num_item

        u_cnt = collections.Counter(user_ids)
        i_cnt = collections.Counter(item_ids)

        self.user_ids = user_ids
        self.item_ids = item_ids

        self.user_map = user_map
        #print(user_map)     #{'1': 0, '2': 1, '3': 2, '4': 3, '5': 4, '6': 5, '7': 6, '8': 7, '9': 8, '10': 9, '11': 10..}
        self.item_map = item_map
        #print(item_map)     #{'32': 0, '23': 1, '28': 2, '38': 3, '25': 4, '37': 5, '4': 6, '8': 7, '48': 8..}
        self.sequences = None
        self.test_sequences = None

        self.min_support=min_support
        self.min_threshold=min_threshold

    def __len__(self):

        return len(self.user_ids)

    def tocoo(self):
        """
        Transform to a scipy.sparse COO matrix.
        """

        row = self.user_ids
        col = self.item_ids
        data = np.ones(len(self))

        return sp.coo_matrix((data, (row, col)),
                             shape=(self.num_users, max(col)+1))

    def tocsr(self):
        """
        Transform to a scipy.sparse CSR matrix.
        """

        return self.tocoo().tocsr()

    def to_sequence(self, sequence_length=5, target_length=1):
        """
        Transform to sequence form.
        Valid subsequences of users' interactions are returned. For
        example, if a user interacted with items [1, 2, 3, 4, 5, 6, 7, 8, 9], the
        returned interactions matrix at sequence length 5 and target length 3
        will be be given by:
        sequences:
           [[1, 2, 3, 4, 5],
            [2, 3, 4, 5, 6],
            [3, 4, 5, 6, 7]]
        targets:
           [[6, 7],
            [7, 8],
            [8, 9]]
        sequence for test (the last 'sequence_length' items of each user's sequence):
        [[5, 6, 7, 8, 9]]
        Parameters
        ----------
        sequence_length: int
            Sequence length. Subsequences shorter than this
            will be left-padded with zeros.
        target_length: int
            Sequence target length.
        """
       
        # change the item index start from 1 as 0 is used for padding in sequences
        for k, v in self.item_map.items():
            self.item_map[k] = v + 1
        self.item_ids = self.item_ids + 1
        self.num_items += 1

        max_sequence_length = sequence_length + target_length

        # Sort first by user id
        sort_indices = np.lexsort((self.user_ids,)) 
        
        user_ids = self.user_ids[sort_indices]
 
        item_ids = self.item_ids[sort_indices]
        
        user_ids, indices, counts = np.unique(user_ids,         #returns uniques user_ids
                                              return_index=True, #user indices
                                              return_counts=True) #counts=number of items a user interacted
        #print([user_ids[:10], indices[:10], counts[:10]])
        num_subsequences = sum([c - max_sequence_length + 1 if c >= max_sequence_length else 1 for c in counts])
        # returns number of subsuqences for all users 
        sequences = np.zeros((num_subsequences, sequence_length),
                             dtype=np.int64)
        sequences_targets = np.zeros((num_subsequences, target_length),
                                     dtype=np.int64)
        sequence_users = np.empty(num_subsequences,
                                  dtype=np.int64)

        test_sequences = np.zeros((self.num_users, sequence_length),
                                  dtype=np.int64)
        test_users = np.empty(self.num_users,
                              dtype=np.int64)

        _uid = None
        for i, (uid,
                item_seq) in enumerate(_generate_sequences(user_ids,
                                                           item_ids,
                                                           indices,
                                                           max_sequence_length)):
            if uid != _uid:
                test_sequences[uid][:] = item_seq[-sequence_length:]      # creating a test set of users+seq
                                                                      #test set = the last possible subsequence 
                test_users[uid] = uid
                #print(test_users[:5])                array (from 0 to num_users) 
                _uid = uid
                #print(test_sequences)          # [ 39  40  41  42  43]
                                                # [136 137 138 139 140]
                                                # [163 164 165 166 167] 
                                              #for one user_id test sequence is taken 
                                              #  based on the last N(=sequence_length) values of the first subsequence
                                              # e.g. for user 0 first subseq is [36 37 38 39 40 41 42 43], so test sequence
                                              #will be  [ 39  40  41  42  43]. Each user has 1 test sequence
            sequences_targets[i][:] = item_seq[-target_length:] 
                                                                              #separate targets:
                                                                             #    [41 42 43]
                                                                              #   [40 41 42]
                                                                               #  [39 40 41]
            sequences[i][:] = item_seq[:sequence_length]                  #from train sequences
                                                                          # [36 37 38 39 40]
                                                                          # [35 36 37 38 39]
                                                                          # [34 35 36 37 38] 
            sequence_users[i] = uid

        self.sequences_o = SequenceInteractions(sequence_users, sequences, sequences_targets)
        self.test_sequences_o = SequenceInteractions(test_users, test_sequences)

        rules=mine_rules(seq=sequences, min_support=self.min_support, min_threshold=self.min_threshold)
        sequences_filtered_train = freq_item(sequences, rules)
        sequences_filtered_test= freq_item (test_sequences, rules)
        self.sequences = SequenceInteractions(sequence_users, sequences_filtered_train, sequences_targets) #see _init_ from SeqInt
                                                      # it sets self.users/sequences/targets/L/T variables 
     
        self.test_sequences = SequenceInteractions(test_users, sequences_filtered_test)
        


class SequenceInteractions(object):
    """
    Interactions encoded as a sequence matrix.
    Parameters
    ----------
    user_ids: np.array
        sequence users
    sequences: np.array
        The interactions sequence matrix, as produced by
        :func:`~Interactions.to_sequence`
    targets: np.array
        sequence targets
    """

    def __init__(self,
                 user_ids,
                 sequences,
                 targets=None):
        self.user_ids = user_ids
        self.sequences = sequences
        self.targets = targets

        self.L = sequences.shape[1]
        self.T = None
        if np.any(targets):
            self.T = targets.shape[1]


def _sliding_window(tensor, window_size, step_size=1):  # breaking sequence down to subsequences. 
    if len(tensor) - window_size >= 0:
        for i in range(len(tensor), 0, -step_size):
            if i - window_size >= 0:
                yield tensor[i - window_size:i]
            else:
                break
    else:
        num_paddings = window_size - len(tensor)
        # Pad sequence with 0s if it is shorter than windows size.
        yield np.pad(tensor, (num_paddings, 0), 'constant')


def _generate_sequences(user_ids, item_ids,
                        indices,
                        max_sequence_length):
    for i in range(len(indices)):

        start_idx = indices[i]

        if i >= len(indices) - 1:
            stop_idx = None
        else:
            stop_idx = indices[i + 1]

        for seq in _sliding_window(item_ids[start_idx:stop_idx],
                                   max_sequence_length):
            yield (user_ids[i], seq)   #returns sequences for each user, size: (N_users * N_seq)x  max_sequence_length
                                        # 0 [36 37 38 39 40 41 42 43]
                                        # 0 [35 36 37 38 39 40 41 42]
                                        # 0 [34 35 36 37 38 39 40 41]
                                        # 0 [33 34 35 36 37 38 39 40]


