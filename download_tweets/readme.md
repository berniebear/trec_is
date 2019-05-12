### For V3 version

First, you need to download the **TREC-IS-Client-v3.jar** file and **info.json** file using the two buttons on [here]([http://dcs.gla.ac.uk/~richardm/TREC_IS/2019/2019_Instructions.html](http://dcs.gla.ac.uk/~richardm/TREC_IS/2019/2019_Instructions.html)) and put them in a folder together. Second, you need to open the info.json file and edit the information in here for your particular institution:

- **institution** should be the name of your company or university.
- **contactname** should be the name of the person downloading the dataset.
- **email** should be the email address of that person.
- **type** should be either 'academic', 'public sector', or 'industry'.
- **request** should be the dataset identifier, i.e. either 'trecis2018-test', 'trecis2018-train' or 'trecis2019-A-test'. If you are reading this page, you probably want 'trecis2019-A-test', although you can also download the older events to use as training data (if you want to download all of them, you will need to call the client jar three times, with a different 'request' line in the info.json file each time).

Once you have finished editing the info.json file, save it and then run the following command from a shell or terminal:

```
java -jar TREC-IS-Client-v3.jar info.json
```



### For V2 version

Because some tweets in the test data cannot be accessed by the official API (their accounts have been suspended).
Here we pick up the second method, which is downloading the archived tweets contents by the API provided by TREC.

You can download the jar file from the [official website](http://dcs.gla.ac.uk/~richardm/TREC_IS/2018/2018TestDataset.html)
Then run the script

```bash
bash download_tweets.sh
```
Notice the jar file has been modified this year, and it will automatically download all topics in test/train as the .gz file, which you need to uncompress manually.

Then, we can use the `merge_tweets.py` in this folder to merge them and convert them to the same format as the official API.


