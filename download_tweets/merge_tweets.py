import os
import json


def merge_tweets_v3():
    """
    It merges all training data into a file and testing data into another file for 2019-B contest.
    We maintain most things the same as `merge_tweets_v2`.
    Note that to avoid breaking things we built for 2019-A, we will keep the output filenames the same, such as
        `all-tweets.txt` and `all-tweets-2019.txt`, but note that they represent the training and testing tweets.
    """
    filename_list = []
    for filename in os.listdir('.'):
        if filename.startswith("trecis") and filename.endswith(".json"):
            filename_list.append(filename)
    filename_list = sorted(filename_list)

    formatted_tweet_list_train = []
    formatted_tweet_list_test = []
    count_inconsistent = 0
    for filename in filename_list:
        with open(filename, 'r', encoding='utf8') as f:
            for line in f:
                try:
                    content = json.loads(line)
                except:
                    print(filename)
                    quit()
                formatted_content = json.loads(content['allProperties']['srcjson'])
                formatted_content['full_text'] = formatted_content['text']

                if 'entities' not in formatted_content:
                    count_inconsistent += 1
                    entities = dict()
                    entities["symbols"] = formatted_content['symbolEntities']
                    entities["urls"] = formatted_content['urlEntities']
                    entities["hashtags"] = formatted_content['hashtagEntities']
                    entities["user_mentions"] = formatted_content['userMentionEntities']
                    entities["media"] = formatted_content['mediaEntities']
                    # To make the "start" and "end" API consistent with others
                    for entity_name in ["hashtags", "user_mentions", "urls"]:
                        for iEntity, entity in enumerate(entities[entity_name]):
                            entity['indices'] = [entity['start'], entity['end']]
                            entities[entity_name][iEntity] = entity
                    formatted_content['entities'] = entities
                    # Some other API convert
                    formatted_content['retweet_count'] = formatted_content['retweetCount']
                    formatted_content['favorite_count'] = formatted_content['favoriteCount']
                    formatted_content['user']['favourites_count'] = formatted_content['user']['favouritesCount']
                    formatted_content['user']['followers_count'] = formatted_content['user']['followersCount']
                    formatted_content['user']['statuses_count'] = formatted_content['user']['statusesCount']
                    formatted_content['user']['geo_enabled'] = formatted_content['user']['isGeoEnabled']
                    formatted_content['user']['verified'] = formatted_content['user']['isVerified']
                    formatted_content['user']['listed_count'] = formatted_content['user']['listedCount']
                    formatted_content['user']['friends_count'] = formatted_content['user']['friendsCount']

                if filename.startswith("trecis2019-B"):
                    formatted_tweet_list_test.append(formatted_content)
                else:
                    formatted_tweet_list_train.append(formatted_content)

    if count_inconsistent > 0:
        print("There are {} tweets have inconsistent API about the entities, "
              "and they are automatically converted.".format(count_inconsistent))
    print("There are {0} tweets for training and {1} tweets for testing".format(
        len(formatted_tweet_list_train), len(formatted_tweet_list_test)))

    outfile = '../data/all-tweets.txt'
    with open(outfile, 'w', encoding='utf8') as fout:
        for tweet in formatted_tweet_list_train:
            fout.write(json.dumps(tweet) + '\n')

    outfile = '../data/all-tweets-2019.txt'
    with open(outfile, 'w', encoding='utf8') as fout:
        for tweet in formatted_tweet_list_test:
            fout.write(json.dumps(tweet) + '\n')


def merge_tweets_v2():
    """
    For merging tweets downloaded by TREC-IS-Client-v2.jar, and used for 2019-A contest.
    It is compatible with the tweets downloaded by TREC-IS-Client-v3.jar, because the downloaded data format is same.
    Note we cannot merge all tweets of 2018 and 2019 together, because we should assume the test data is invisible,
        so the PCA and some other things could be trained on training data and appiled on testing data.
    For some tweets in 2019-test, different entities are in different keys, not sure if it dues to twitter API.
    :return:
    """
    filename_list = []
    for filename in os.listdir('.'):
        if filename.endswith(".json"):
            filename_list.append(filename)
    filename_list = sorted(filename_list)

    formatted_tweet_list = []
    formatted_tweet_list_2019 = []
    count_inconsistent = 0
    for filename in filename_list:
        with open(filename, 'r', encoding='utf8') as f:
            for line in f:
                content = json.loads(line)
                formatted_content = json.loads(content['allProperties']['srcjson'])
                formatted_content['full_text'] = formatted_content['text']

                if 'entities' not in formatted_content:
                    count_inconsistent += 1
                    entities = dict()
                    entities["symbols"] = formatted_content['symbolEntities']
                    entities["urls"] = formatted_content['urlEntities']
                    entities["hashtags"] = formatted_content['hashtagEntities']
                    entities["user_mentions"] = formatted_content['userMentionEntities']
                    entities["media"] = formatted_content['mediaEntities']
                    # To make the "start" and "end" API consistent with others
                    for entity_name in ["hashtags", "user_mentions", "urls"]:
                        for iEntity, entity in enumerate(entities[entity_name]):
                            entity['indices'] = [entity['start'], entity['end']]
                            entities[entity_name][iEntity] = entity
                    formatted_content['entities'] = entities
                    # Some other API convert
                    formatted_content['retweet_count'] = formatted_content['retweetCount']
                    formatted_content['favorite_count'] = formatted_content['favoriteCount']
                    formatted_content['user']['favourites_count'] = formatted_content['user']['favouritesCount']
                    formatted_content['user']['followers_count'] = formatted_content['user']['followersCount']
                    formatted_content['user']['statuses_count'] = formatted_content['user']['statusesCount']
                    formatted_content['user']['geo_enabled'] = formatted_content['user']['isGeoEnabled']
                    formatted_content['user']['verified'] = formatted_content['user']['isVerified']
                    formatted_content['user']['listed_count'] = formatted_content['user']['listedCount']
                    formatted_content['user']['friends_count'] = formatted_content['user']['friendsCount']

                if filename.startswith("trecis2019"):
                    formatted_tweet_list_2019.append(formatted_content)
                else:
                    formatted_tweet_list.append(formatted_content)

    if count_inconsistent > 0:
        print("There are {} tweets have inconsistent API about the entities, "
              "and they are automatically converted".format(count_inconsistent))
    print("There are {0} tweets for 2018 and {1} tweets for 2019".format(
        len(formatted_tweet_list), len(formatted_tweet_list_2019)))

    outfile = '../data/all-tweets.txt'
    with open(outfile, 'w', encoding='utf8') as fout:
        for tweet in formatted_tweet_list:
            fout.write(json.dumps(tweet) + '\n')

    outfile = '../data/all-tweets-2019.txt'
    with open(outfile, 'w', encoding='utf8') as fout:
        for tweet in formatted_tweet_list_2019:
            fout.write(json.dumps(tweet) + '\n')


def merge_tweets_v1():
    """
    It is used for merging the tweets downloaded by TREC-IS jar v1 version.
    However, as they update the jar file for 2019, and the format of the data is changed, so this function is deprecated
    :return:
    """
    formatted_tweet_list = []
    for filename in os.listdir('.'):
        if filename.endswith(".json"):
            with open(filename, 'r', encoding='utf8') as f:
                for line in f:
                    formatted_content = dict()
                    content = json.loads(line)
                    formatted_content['id_str'] = content['identifier']
                    formatted_content['full_text'] = content['text']

                    metadata = content['metadata']
                    entities = dict()
                    for feature_name in ['hashtags', 'symbols', 'user_mentions', 'urls']:
                        attribute_name = 'entities.{}'.format(feature_name)
                        if attribute_name in metadata:
                            entities[feature_name] = json.loads(metadata[attribute_name])
                    entities['media'] = content['media']
                    formatted_content['entities'] = entities

                    formatted_content['created_at'] = metadata['created_at']
                    for feature_name in ['retweet_count', 'favorite_count']:
                        formatted_content[feature_name] = int(metadata[feature_name])
                    formatted_content['user'] = dict()
                    formatted_content['user']['verified'] = metadata['user.verified'] == 'true'
                    if 'coordinates' in metadata and metadata['coordinates'] != 'null':
                        formatted_content['coordinates'] = metadata['coordinates']

                    formatted_tweet_list.append(formatted_content)

    outfile = '../data/tweets-content-merged.txt'
    with open(outfile, 'w', encoding='utf8') as fout:
        for tweet in formatted_tweet_list:
            fout.write(json.dumps(tweet) + '\n')


def count_total_line():
    """
    To count all lines in all json files downloaded by the jar file, if it is smaller than 25886, it should be wrong
    :return:
    """
    count = 0
    file_count = 0
    for filename in os.listdir('.'):
        if filename.endswith(".json"):
            file_count += 1
            with open(filename, 'r', encoding='utf8') as f:
                for line in f:
                    count += 1
    print("There are {0} lines in {1} json files".format(count, file_count))


if __name__ == '__main__':
    merge_tweets_v3()
