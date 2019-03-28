Because some tweets in the test data cannot be accessed by the official API (their accounts have been suspended).
Here we pick up the second method, which is downloading the archived tweets contents by the API provided by TREC.

You can download the jar file from the [official website](http://dcs.gla.ac.uk/~richardm/TREC_IS/2018/2018TestDataset.html)
Then run the script

```bash
bash download_tweets.sh
```
Notice the jar file has been modified this year, and it will automatically download all topics in test/train as the .gz file, which you need to uncompress manually.

Then, we can use the `merge_tweets.py` in this folder to merge them and convert them to the same format as the official API.


