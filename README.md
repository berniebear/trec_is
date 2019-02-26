# TREC-IS
TREC Incident Stream shared task (official website [here](http://dcs.gla.ac.uk/~richardm/TREC_IS/))

## Environment

`Python 3.6`

TensorFlow, scikit-learn

As we don't have sudo on server, we use conda to manage all packages.


## Task Description

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

Notice that some tweets are not accessible due to the suspend of the account
- There are 

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
```bash
Information Type Precision (positive class, multi-type, macro): 0.18630789182513743
Information Type Recall (positive class, multi-type, macro): 0.08473317660637916
Information Type F1 (positive class, multi-type, macro): 0.1031639919206761
Information Type Accuracy (overall, multi-type, macro): 0.8991002830570157
Priority Estimation Error (mean squared error, macro): 0.09434289369225503
Information Type Precision (any valid type, micro): 0.3874517740813358
Information Type Recall (any valid type, micro): 0.6394572025052192
Information Type F1 (any valid type, micro): 0.48253318104840304
Information Type Accuracy (any valid type, micro): 0.335877476748888
```

## Todo
- Add more features (fasttext, combine word2vec with tfidf, BERT)
- Add feature for each event
- Use a better method to predict the importance score

## Reference

Official Baseline system: https://github.com/cbuntain/trecis

It is quite useful as it includes many features and pre-trained vectors and introduce another large dataset that could be used
