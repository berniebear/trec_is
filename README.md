## Multimodl Social Media Fitlering Project
The is the codebase for the following two papers:
1. "Multimodal Filtering of Social Media for Temporal Monitoring and Event Analysis", Po-Yao Huang, Junwei Liang, Jean-Baptiste Lamare, Alexander G Hauptmann, ICMR 2018.

2. CMU-Informedia at TREC 2019 Incident Streams Track, Junpei Zhou Xinyu Wang, Po-Yao Huang, Alexander Hauptmann, TREC 2019.

## Environment

`Python 3.6`

`scikit-learn==0.19.2`

`TensorFlow==1.12.0` (not necessary for running the main.py, but needed for extracting features by `BERT` and `Skip-thought`, please refer to `feature_tools` folder for details)

`spacy==2.0.16` with `python -m spacy download en_core_web_sm`.

If you don't have sudo on server, you can use conda to manage all packages.

## Dataset and Task Description

1. TREC Incident Stream shared task (official website [here](http://dcs.gla.ac.uk/~richardm/TREC_IS/))

2. Social Media Tweet Filtering for Las Vegas Shooting and Boston Marathon Marathon Bombing (Dataset: [here](http://dcs.gla.ac.uk/~richardm/TREC_IS/))


### How to Run

1. Prepare tweets by following the steps in `download_tweets/readme.md`, 
and remember to change `merge_tweets_v3` to `merge_tweets_v2` if you want to tune pararmeters on 2019-A data.

2. Uncomments the `quit()` in line 89 which will prepare those four files including `tweet_text_out_file` 
in `out` folder and then quit. You will see some output message like
    ```
    The tweet text and ids has been written to out/tweets-clean-text.txt and out/tweets-id.txt
    The tweet text and ids has been written to out/tweets-clean-text-2019.txt and out/tweets-id-2019.txt
    ```

3. Manually run the `extract_features.sh` in `feature_tools` which will prepare several npy feature matrix in `out` folder.

4. Comments the `quit()` in line 89 and run the script `main.py` with `predict_mode`, such as
    ```
    python main.py --class_weight_scheme customize --predict_mode --use_pca --model rf --additional_weight 0.2 --force_retrain --train_regression
    ```
    Here the `--train_regression` will not influence the result, because in `predict_mode` it will only write the regression result to file for further usage (such as `get_submission`)
    We can use `--model` to choose models (usually use naive bayes or random forest).
    Use `--event_wise` to train each model for each event.
    Use `--normalize_feat` for normalizing all features. (For example, it normalize the BERT feature on axis=1 and 
    normalize the glove feature on axis=1, and then concatenate those two features).
    Please refer to `options.py` to get more info about different parameters.

5. After running step 4, there will be model checkpoint file stored, and then we can run `main.py` with `get_submission`
such as
    ```
    python main.py --class_weight_scheme customize --get_submission --use_pca --force_retrain --model rf --additional_weight 0.2 --merge_priority_score simple
    ```
    We can use `--merge_priority_score` to control if we want to use `simple` strategy or `advanced` strategy (need to specify `advanced_predict_weight`).
    The default method is to pick top 2 for each prediction. You can change it according to the paramters shown in `options.py` (under the comment `# For final submission`)

6. If you are tuning parameters on 2019-A, you can run `python evaluate_v3.py --weight 10.0` to get the results.
The parameter tuning results are recorded [here](https://docs.google.com/spreadsheets/d/1mqCOAFt4X4GDwjtU4ggcYkeKuMYvFiFTrN4UtmJo8Bs/edit?usp=sharing).

7. If you want to run the ensemble, you need to add `--cross_validate` in all previous commands, for example:
    ```bash
    python main.py --model rf --cross_validate --predict_mode
    python main.py --model rf --cross_validate --get_submission
    ```
    For ensemble, you need to check that all files in `ensemble_dir` with `dev_predict_` prefix and `test_predict_` prefix
    are the files we want. If there are files generated by other settings' models, they will make the ensemble worse.
    Currently, we don't use ensemble for our submission, so `--train_regression` has not been implemented for ensemble.

8. If you want to tune hyper-parameters, you can use `--cross_validate`. For example:
    ```
    python main.py --cross_validate --use_pca --search_best_parameters --random_search_n_iter 100
    ```

**Before submission**

Use the `awk` to replace the last field for each submission and then gzip it for submission.
```bash
awk -F'\t' 'BEGIN { OFS="\t" } {$7="Informedia-rf1"} 1' submission_rf_1 > submission_rf_1.run
gzip submission_rf_1.run
```


** Submission Command for 2019-A **
```
python main.py --class_weight_scheme customize --predict_mode --use_pca --force_retrain --model rf
python main.py --class_weight_scheme customize --get_submission --use_pca --force_retrain --model rf
```

** Submission Command for 2019-B **

Recorded [here](https://docs.google.com/spreadsheets/d/1mqCOAFt4X4GDwjtU4ggcYkeKuMYvFiFTrN4UtmJo8Bs/edit?usp=sharing).
with the submission row highlighted.

## Task Description

### [Important] Change in 2019 compared with 2018
[Here](http://dcs.gla.ac.uk/~richardm/TREC_IS/2019/2019Changes.html) is a page to describe the differents.

### Event Types
wildfire, earthquake, flood, typhoon/hurricane, bombing, shooting

### Task
There is only a single task for the first year of the track (2018): classifying tweets by information type (high-level).

The goal of this task is for systems to categorize the tweets in each event/incident's stream into different information feeds.

the task aim is to assign ontology labels (information types) to each tweet within the event stream.

As noted above, the ontology has multiple layers, moving from generic information types to the
very specific. For this reason, we denote information types as either `top-level intent’, ‘high-level’
or ‘low-level’. For example, a top-level intent might be ‘Reporting’ (the user is reporting some
information). Within reporting, a high-level type might be ‘Service Available’ (the user is
reporting that some service is being provided). Within service available, a low-level type might
be ‘Shelter Offered’ (shelter is offered for affected citizens).

This task is Classifying Tweets by Information Type (high-level). I.e. the goal is to categorize
tweets into the information types listed as high-level. One category per tweet.

### Submission format
The submission format is slightly different from the TREC-eval format, and you can find the requirement [here](http://dcs.gla.ac.uk/~richardm/TREC_IS/TREC_2018_Incident_Streams_Guidelines.pdf)

## Data

`ITR-H.types.v2.json` Contains the high-level label which we need to predict for tweets.

`TRECIS-CTIT-H-Training.json` Contains the training data
  - `postID` is the tweet id and we can use it to get tweet text and img
  - `categories` is the category we want to classify
  - `priority` will be mapped to score (will it incluence the evaluation?)
  - `indicatorTerms` are some important terms extracted by the annotator

`TRECIS-CTIT-H-Test.topics` contains 15 topics (events) for test data

`TRECIS-CTIT-H-Test.tweetids.tsv` Contains the Tweets stream for each topic in `TRECIS-CTIT-H-Test.topics`

`TRECIS-2018-TestEvents-Labels` This folder contains the labels of those test events

Tweet ids and json contents (generated by `extract_train_ids_to_file` in `utils.py`)
  - `train-ids.txt` Contains tweets identifiers for training data
  - `test-ids.txt` Contains tweets identifiers for test data
  - `train-tweets.txt` Contains tweets contents for training, each line is a json format
  - `test-tweets.txt` Contains tweets contents for test, each line is a json format

### Statistics Info
25 label categories

Train: 1,335 instances, including 6 events

Test: 22,216 instances, including 15 events

Notice that some tweets are not accessible by Twitter API due to the suspend of the account (96 tweets in train and 1263 in test), but still accessible by the jar provided by TREC-IS host.
## Get Twitter according to IDs
These datasets will be distributed as a list of tweet identifiers for each incident. Participants will
need to fetch the actual JSON tweets using publicly available tools, such as twarc
(https://github.com/DocNow/twarc), the TREC Microblog Track twitter-tools
(https://github.com/lintool/twitter-tools), or any other tool for crawling twitter data.

For twarc tools, you can use this command to get tweets from a file contains ids
```bash
twarc hydrate ids.txt > tweets.jsonl
```

Actually, we can use `twitter.com/user/status/<TWEETID>` and then it will redirect to the true page.
But it requires too many effort to get the json (need to use scrapy and extract json from a dynamic webpage).

To make it easier, we need a developer account on `https://developer.twitter.com/en.html`

Finally, I get Twitter developer account by my Andrew Email, the passwd is generated by Chrome.

The credentials for JunpeiZhou have been saved to your configuration file at /Users/jpzhou/.twarc

```bash
Consumer API keys
yy0XFA7LsdBWiJ0NMLl92JVdp (API key)
bBtK6rk2wf4VTejobPVtARbn5IKPtMPfl02pVSx9lDCgty6nYJ (API secret key)

Access token & access token secret
1096268271286931459-l5CjMwD2VhaliVt811oL80BBXJfzSm (Access token)
IeR7tyuzfyubrZUlnAT5sJL0eLX3o8BljMPO53YoPwv5H (Access token secret)
Read and write (Access level)
```

## Current Result
Baseline (only with hand-crafted features)
```
Information Type Precision (any valid type, micro): 0.3874517740813358
Information Type Recall (any valid type, micro): 0.6394572025052192
Information Type F1 (any valid type, micro): 0.48253318104840304
Information Type Accuracy (any valid type, micro): 0.335877476748888
```

Add fasttext (avg)
```
Information Type Precision (any valid type, micro): 0.4057043435033866
Information Type Recall (any valid type, micro): 0.6691606026442554
Information Type F1 (any valid type, micro): 0.5051450676982592
Information Type Accuracy (any valid type, micro): 0.3534169025475131
```

Add BERT
```
Information Type Precision (any valid type, micro): 0.4143519950510362
Information Type Recall (any valid type, micro): 0.6791725816264449
Information Type F1 (any valid type, micro): 0.5146962769431743
Information Type Accuracy (any valid type, micro): 0.3615547917509098
```

Add skip-thought
```
Information Type Precision (any valid type, micro): 0.42567274976801733
Information Type Recall (any valid type, micro): 0.6886509207365893
Information Type F1 (any valid type, micro): 0.5261306724777306
Information Type Accuracy (any valid type, micro): 0.3734836231298019
```

Change to data got by jar (less tweets missing)
```
Information Type Precision (any valid type, micro): 0.43939964832956546
Information Type Recall (any valid type, micro): 0.681371116953939
Information Type F1 (any valid type, micro): 0.5342648799297522
Information Type Accuracy (any valid type, micro): 0.38339061868176305
```

Currently rank 8/40 in 2018 leaderboard 


**Experiments to see if cross-val is comparable with previous setting**

2018-train + test perform Cross-validation F1: 0.56, Accuracy: 0.39 (so it is comparable). Here are details
```
2019-04-02 22:53:14,006 - root - INFO - The acc score in cross validation is [0.4090368608799049, 0.38002378121284186, 0.38882282996432815, 0.38106565176022833, 0.3946241674595623]
2019-04-02 22:53:14,006 - root - INFO - The f1 score in cross validation is [0.5778834720570749, 0.5491959190731454, 0.5579635362917097, 0.5509147393855712, 0.5642869371682931]
2019-04-02 22:53:14,006 - root - INFO - The average acc score is 0.3907146582553731
2019-04-02 22:53:14,006 - root - INFO - The average f1 score is 0.5600489207951589
```
2018-train and then test on cross-validation (however, it has a kind of leak, because the train is included in the cross-validation test data). The average acc score is `0.40` and the average f1 score is `0.55`.

When use **late fusion**, the f1 score will drop a lot (not sure if it dues to the method to do late fusion is not tuned).

When use **random forest** `--cross_validate --model rf` with `['hand_crafted', 'fasttext', 'skip_thought', 'bert', 'glove']`
```
2019-04-11 18:40:46,411 - root - INFO - The average accuracy score is 0.6419636810421194
2019-04-11 18:40:46,411 - root - INFO - The average precision score is 0.6434336557411932
019-04-11 18:40:46,411 - root - INFO - The average recall score is 0.9848954373241163
2019-04-11 18:40:46,411 - root - INFO - The average f1 score is 0.7783424367579415
```

After adding `fasttext_crawl` the performance increase a little bit

```
2019-04-12 01:36:05,115 - root - INFO - The average accuracy score is 0.6432953271387535
2019-04-12 01:36:05,116 - root - INFO - The average precision score is 0.6444531168264113
2019-04-12 01:36:05,116 - root - INFO - The average recall score is 0.9854160962884855
2019-04-12 01:36:05,116 - root - INFO - The average f1 score is 0.7792608064917799
```

After adding `hashtag` the performance increase a little bit
```
2019-04-14 19:37:33,258 - root - INFO - The average accuracy score is 0.647148392731683
2019-04-14 19:37:33,258 - root - INFO - The average precision score is 0.6484004516753854
2019-04-14 19:37:33,258 - root - INFO - The average recall score is 0.985834526938896
2019-04-14 19:37:33,259 - root - INFO - The average f1 score is 0.7822509304825445
```

After setting `no_class_weight` the f1 drops to `0.7758`, so we had better always use `balanced` for the class weight.

Using event-wise model (naive bayes model with `['hand_crafted', 'fasttext', 'skip_thought', 'bert', 'glove', 'fasttext_crawl']`)
```
{'accuracy': 0.4555503209082257, 'precision': 0.46044030790481144, 'recall': 0.9512387607178248, 'f1': 0.6145547249635873}
```

Using event-wise model (random forest with `['hand_crafted', 'fasttext', 'skip_thought', 'bert', 'glove', 'fasttext_crawl', 'hashtag']`)
```
The final evaluation metrics val for event-wise model is {'accuracy': 0.6712643466219029, 'precision': 0.6720118180223713, 'recall': 0.9788352416057818, 'f1': 0.7960859213911053}
```

#### The following is all about NB model

After searching parameter for NB with `['hand_crafted', 'fasttext-avg', 'skip-thought', 'bert', 'glove-avg']`
```
F1: 0.6709
{'alpha': 0.848978980643246, 'binarize': 0.30859955544919404, 'fit_prior': True}
```
Use `event_wise` for NB with those parameters get `0.7047` F1.
```
2019-04-18 15:04:57,564 - root - INFO - The final evaluation metrics val for event-wise model is {'accuracy': 0.5564857536983304, 'precision': 0.5606054867072, 'recall': 0.96216209768519, 'f1': 0.7047413283261292}
```

Using `['hand_crafted', 'fasttext-avg', 'skip-thought', 'bert', 'glove-avg', 'fasttext-crawl', 'hashtag]` is `0.6040`.
Using `['hand_crafted', 'fasttext-avg', 'skip-thought', 'bert', 'glove-avg', 'fasttext-crawl']` is `0.6722`.
(so in following experiments we will discard the `hashtag` feature)

- Using `PCA=100` we get `0.6909` (after testing PCA=50, 100, 150, we find 100 performs best)

- After doing leave-one-out feature selection, I decide to keep those features: `['hand_crafted', 'fasttext-avg', 'skip-thought', 'bert', 'glove-tfidf', 'fasttext-crawl']`.
- The current performance with bernoulli NB is `0.6768`, after adding `--pca` it achieves `0.6936`
- With `--use_pca` and `--event_wise` it can reach `0.7282`.

- After adding normalization, no matter with PCA or not, the performance drops a lot (without pca it is `0.6058`, with pca it is `0.6590`) So we may not use normalization.
- After adding `late_fusion` the performance drops (from `0.6936` to `0.6368`) So we may not use late fusion.


#### The following is all about linearSVC model
Use the feature selected according to NB: `['hand_crafted', 'fasttext-avg', 'skip-thought', 'bert', 'glove-tfidf', 'fasttext-crawl']`
- `--use_pca`, the best f1 is `0.6934`
    ```
    The best parameter is `{'C': 0.1, 'class_weight': 'balanced', 'dual': False, 'penalty': 'l2'}`
    ```
- `--normalize_feat` but without PCA, f1 reaches `0.7496`
- For `svm_rbf` with pca, f1 is `0.7085`, without PCA it will raise Memory error, and running time will be too long. So we will keep using linear SVM instead of kernel-based SVM.
- `--use_cpa` and `--event_wise` get f1 `0.7201`
- `--normalize_feat` and `--event_wise` get f1 `0.7691`


#### The following is all about Random Forest model
Use the feature selected according to NB: `['hand_crafted', 'fasttext-avg', 'skip-thought', 'bert', 'glove-tfidf', 'fasttext-crawl']`
- `--use_pca` can reach `0.7832`.
- Without pca the f1 is `0.7775`, which also takes much longer time, so we had better use pca.
- `--use_pca` and random search parameter can reach f1 `0.7845`:
    ```
    The best parameter is {'criterion': 'gini', 'max_depth': 64, 'max_features': 213, 'min_samples_leaf': 5, 'min_samples_split': 43, 'n_estimators': 128, 'class_weight': 'balanced', 'n_jobs': -1}
    ```
- `--use_pca` with `--event_wise` can reach f1 `0.7957`


#### The following is all about xgboost model
- Use all default parameter with pca f1 is `0.7739`
- `--use_pca` and random search parameter can reach f1 `0.8063`
    ```
    The best parameter is {'subsample': 0.9, 'n_jobs': -1, 'n_estimators': 500, 'max_depth': 8, 'learning_rate': 0.05, 'gamma': 0, 'colsample_bytree': 0.9}
    ```
- `--use_pca` with `--event_wise` can reach f1 `0.8117`


#### The following is all about ensemble
- Ensemble of nb (`0.6936`) and svm_linear (`0.6934`) with svm model can get f1 `0.7629`.

#### Some other notes
When we use the KFold of sklearn, we get the weighted average ratio around `3.95`.
When we implement the stratified K-fold based on a paper published on 2011, the ratio is around `4.00`.
It means the stratified method is really better than K-folder, but the difference is not so obvious.

## 2019-B Todo
- Adjust parameters for type classification according to the new evaluation script.
For example, use class_weight_scheme == 'customize' which gives more weights to actionable classes.
And the additional weights added in the customized class weights could also be tuned.
- [done] Regression to get priority score.
- Classification to get priority score (give more weights to the high priority).
- Run the feature extraction tools again on all data (including 2019-B) and predict.
- For regression, change it to classification and give higher weights to high priority.

** Some Experimental Results **
Use random forest (customized weight).

When >0.7 copy class score, otherwise average regression score and class score (already better than most teams).
```
Priority Estimation Error (mean squared error, macro): 0.06123941348506185
Priority Estimation Error High Importance (mean squared error, macro): 0.1381861719212382
```

When >0.7 copy class score, otherwise use regression score (almost the best among all teams).
```
Priority Estimation Error (mean squared error, macro): 0.053614565907744216
Priority Estimation Error High Importance (mean squared error, macro): 0.105253727545082
```

When only use the class score (Simple method), and set additional_weight to 1.0 (quite good in all aspects)
However, it actually has a bug, which is that some scores in submission file is larger than 1.0, and it "fools" the evaluation script.
```
High Importance Alert Worth: -0.16628242074927946
Accumulated Alert Worth: -0.2589881529385931
Information Type F1 (positive class, multi-type, macro): 0.1153338898176881
High Importance Information Type F1 (positive class, multi-type, macro): 0.057893041357435455
Priority Estimation Error (mean squared error, macro): 0.0848325638255392
Priority Estimation Error High Importance (mean squared error, macro): 0.14055919115397889
```

When set additional_weight to 1.0, --train_regression
```
High Importance Alert Worth: -0.9230547550432279
Accumulated Alert Worth: -0.470109165374908
Priority Estimation Error (mean squared error, macro): 0.07529813107453287
Priority Estimation Error High Importance (mean squared error, macro): 0.1478957047061434
```

## 2019-A Todo
- Submit four models: class_weight_scheme balanced/customize + pick up threshold / top 2
- Round-trip translation / paraphrase (SemEval 2015 Task 1) to do data augmentation
- First classify the higher level, and then classify the target (Request-GoodsService, Request-SearchAndRescue)
- Use a better method to predict the importance score
- Use generative model, try to model the joint distribution of p(x,y) and can extract the feature to feed into the descriptive model (similar to ELMo)
- Borrow idea from Twitter Sentiment Analysis (including pre-processing methods)
- Suggested by Xin: augment interest profile with calling Google API. By the way, remember to deduplicate the tweets (removing tweets of retweet and very similar tweets).
- Relationship between events (domain adaption/shift)
- Use time to predict the priority of tweets, and give high priority tweets more weights in train
- Use indicator terms (such as forming a bag of words) in 2018 training data, because it is a quite important label provided by human labelers
- Retrain skip-thought on additional data

### May 14 Discussion

Junpei:

- [done] cope with rf class_weight (because in onevsrest setting there are only two classes)
- Read new post in Google group about the importance of different categories
- Check inference code of event-wise (and why the score always 0.04)
- [done] Use class weight to retrain the model
- Pick different thresholds for different classes

### May 7 Discussion

Junpei:

- [done] Wait for 2019 test as well as metric release and adjust the model
- [done] BERT use CLS and multi-layer

### Apr 26 Discussion

Junpei:

- [done] Get event-wise result for random forest and XGBoost
- [done] Get ensemble result (use **Weighted Majority Algorithm** as a simple baseline)

### Apr18 Discussion

Junpei:
- [done] Implement a stratified sampling method for multi-label setting.
- [done] Transfer the Parameter Search to sklearn API
- [done] Try late fusion for current new parameters (not good)
- [done] Normalization features before concatenate them
- [done] Leave one out to select features (the `fasttext-tfidf` and `glove-avg` and `hashtag` are removed)
- [done] Add models of linear svm and XGBoost
- [half-done] Do model ensemble (Using Probabilities as Meta-Features)

Xinyu:
- plug fasttext trained on Twitter data
- Add other handcrafted feature, read CBNU paper (last year first prize) and adopt some hand-crafted features
- Temporal information feature
- Additional data feature (from the additional data collected)
- image feature classifier


### Apr12 Discussion
Junpei:
- [done] Add Hashtag feature
- [done] Event-wise model. By the way, the event of each test tweet will be informed, so we can manually choose classifier for each event
- [done] Read paper "A SIMPLE BUT TOUGH-TO-BEAT BASELINE FOR SENTENCE EMBEDDINGS" and try the embedding of that method
- [done] Tune models (random search hyper-parameters for nb and rf and (maybe) linear svm)

Xinyu:
- All things not done, move to 'Apr18 Discussion'

### Mar29 Discussion

Junpei:
- [done] Cross-validation to see if it is comparable (question: Current evaluation is any-type, which means only require to have overlap with the test labels, how to perform it in cross-validation)
- [done] To see if the additional test data added into training is helpful (if data is noisy)
- [done] Late fusion (train model and then average, because different features may not in the same scale; However, for graph-based model such as NB the scale will not influence it)
- [done] Check with host if additional data could be used
- [done] Install the CUDA and other dependencies
- [done] Add Glove to feature
- [done] Use fasttext trained on other dataset (currently we use the fasttext trained by tweets provided by the host)

Xinyu:
- News extract feature
- [done] Explore the data to see if the number of images is large enough and useful
- [done] Manually check if data timestep contains useful information
- [done] Natural Disaster Dataset. As shown in the TREC-IS official website, there are six general events: bombing, earthquake, flood, hurricane, wildfire, shooting. 
The description of each type of general event could be found in its corresponding pdf ("Event Type Profiles" in [webpage](http://dcs.gla.ac.uk/~richardm/TREC_IS/)). 

Bernie:
- [done] Get computatioinal resource for this project
- Check the correctness of get_clean_tweet in utils
- Classifier part

## Demo for displaying tweets

The `utils_display.py` provide tools to extract some tweets along with some information that could be displayed
for the demo.
Note that the `rf.run` in that script is the output file of our system, which is the final submission file.

The demo is in another private repo `Display-Tweets`.

## Reference

Official Baseline system: https://github.com/cbuntain/trecis

It is quite useful as it includes many features and pre-trained vectors and introduce another large dataset that could be used

Some addtional tweets data: 
- https://crisislex.org/data-collections.html#CrisisLexT26
- https://crisisnlp.qcri.org/
- Get tweets by hashtags and keywords through Twitter API (like the way to collect retrospective data)
  - Here are some keywords provided by a paper: #pray4victims: Consistencies In Response To Disaster on Twitter (the official baseline get 741,859 tweets by those keywords)

