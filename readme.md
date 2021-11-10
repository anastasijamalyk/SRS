# Modelling Skip-Behaviors in Sequence-Aware Recommender Systems
Sequential Recommender Systems (SRSs) operate on sequences of items as inputs. If the sequences contain noisy items, SRS will produce a sub-optimal performance as noise may dilute the signal captured by the model. The project is aimed to improve the performance of SRS in two ways: 1) to model relationships between non-adjacent items (skip-behaviors) to account for heterogeneous sequences 2) to introduce noise-filtering techniques to mitigate the impact of noise on sequences. In particular, two models are proposed: Sequential Recommender with Association Rules (SRAR) uses assocition rules to filter out noisy data; Sequential Recommender with Attention (SRAttn) integrates an attention mechanism to detect and deactivate the embeddigns of irrelevant items. 

# Requirements
- Python 3
- PyTorch v1.0+

# Training
The default hyperparameters are optimal for the MovieLens dataset. In order to get the reported performance on Gowalla, the following hyperparameters should be changed:

`python train.py SRAR --dataset=gowalla --d=100 --fc_dim=50 --l2=1e-6 --min_support=9e-5 --min_threshold=0.0366`

`python train.py SRAttn --dataset=gowalla --d=100 --fc_dim=50 --l2=1e-6 --num_heads=5`

Last.fm hyperparameters:

`python train.py SRAR --dataset=lastfm --learning_rate=0.002 --d=200 --fc_dim=50 --l2=5e-6 --num_heads=25`

**Important:** for the Last.fm dataset, a restricting condition `if p not in rated` should be removed in `evaluation.py`:

`predictions = [p for p in predictions if p not in rated] `

Since the dataset contains many repetitions, the model is allowed to recommend items that are present in the training set. 

# Datasets 
- The models were trained on three publicly available datasets: MovieLens, Last.fm, and Gowala. 
-  Each entry in a dataset is represented in the form of a triplet *(user−item−rating)*
- Each rating is substituted with 1 to adjust the data to the implicit feedback scenario.
- The data is split into train.txt and test.txt files as follows: 80% of the user’s sequence is dedicated to the training set, and 20% to the test set. The 80/20 split is subsequently performed on the training set to generate the training and validation sets that are used for parameter tuning. 


# Acknowledgements
The project is heavily built on top of [Spotlight](https://github.com/maciejkula/spotlight), [Caser](https://github.com/graytowne/caser_pytorch), and [CosRec](https://github.com/zzxslp/CosRec). Many thanks for their contribution to the community.