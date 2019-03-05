import os
import json


def merge_tweets():
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
    count_total_line()
