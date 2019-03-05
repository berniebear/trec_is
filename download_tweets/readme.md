Because some tweets in the test data cannot be accessed by the official API (their accounts have been suspended).
Here we pick up the second method, which is downloading the archived tweets contents by the API provided by TREC.

You can download the jar file from the [official website](http://dcs.gla.ac.uk/~richardm/TREC_IS/2018/2018TestDataset.html)
Then run the script
```bash
bash download_tweets.sh
```
Notice there are two lines in this bash script, one for downloading the training tweets, and the other is used for the testing tweets.

After downloading finished, there will be fifteen files, and each of them contains tweets for a type of event.
We can use the `merge_tweets.py` in this folder to merge them and convert them to the same format as the official API.


