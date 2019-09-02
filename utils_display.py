""" This script provides some utils for the purpose of displaying tweets on website."""

import json
import os

from utils import get_tweetid_content

# For Named Entity Recognition
import spacy
nlp = spacy.load("en_core_web_sm")


def get_city2latlng(datafile):
    city2latlng = dict()
    with open(datafile, 'r', encoding='utf8') as f:
        for iLine, line in enumerate(f):
            if iLine == 0:
                continue
            content = line.strip().split(',')
            city_name = content[0][1:-1].lower()
            lat, lng = float(content[6][1:-1]), float(content[7][1:-1])
            city2latlng[city_name] = (lat, lng)
    return city2latlng


def get_state2latlng(datafile):
    state2latlng = dict()
    with open(datafile, 'r', encoding='utf8') as f:
        for line in f:
            content = line.strip().split('\t')
            if len(content) != 4:
                continue
            state_abrr_name = content[0].lower()
            state_full_name = content[-1].lower()
            lat, lng = float(content[1]), float(content[2])
            state2latlng[state_abrr_name] = (lat, lng)
            state2latlng[state_full_name] = (lat, lng)
    return state2latlng


def get_info_for_display(tweet_content_list, keep_no_img_tweet=False, keep_no_latlng_tweet=False):
    """
    Parse the raw data, where each line is a json string containing all information for a tweet.
    What we need for display (currently) is: [id, full_text, media, location, created_at].
    Then the latitude and longitude could be derived from the location + full_text location entity recognition.

    :param tweet_content_list: A list of content where each content is a dict load by json.
    :param keep_no_img_tweet: If we need to keep the tweet that without the image.
    :param keep_no_latlng_tweet: If we need to keep the tweet that without the latitude and longitude information.

    :return: A list of dict, which could be written as a json file later.
    """
    def get_coordinate(location2latlng, location, full_text):
        """If the location cannot be parsed as a valid location, we extract location entity from full_text."""
        result_latlng = None
        # Try to find the lat and lng from the location string.
        loc_list = location.strip().split(',')
        for loc in loc_list:
            loc = loc.strip().lower()
            if loc in location2latlng:
                result_latlng = location2latlng[loc]
        if result_latlng is not None:
            return result_latlng
        # Try to find the lat and lng from the GPE entity in text string.
        doc = nlp(full_text, disable=["tagger", "parser"])
        for ent in doc.ents:
            if ent.label_ == 'GPE':
                loc = ent.text.lower()
                if loc in location2latlng:
                    result_latlng = location2latlng[loc]
                    break
        return (None, None) if result_latlng is None else result_latlng

    city2latlng = get_city2latlng("data/UScity2latlong.csv")
    state2latlng = get_state2latlng("data/USstate2latlong.csv")
    location2latlng = {**city2latlng, **state2latlng}
    location_missing_count = 0
    result = []
    for content in tweet_content_list:
        key2val = dict()
        key2val["id"] = content["id_str"]
        key2val["full_text"] = content["full_text"]

        # For those tweets which have images.
        if "media" in content["entities"] and len(content["entities"]["media"]) > 0:
            if "media_url" in content["entities"]["media"][0]:
                key2val["media"] = content["entities"]["media"][0]["media_url"]
            elif "mediaURL" in content["entities"]["media"][0]:
                key2val["media"] = content["entities"]["media"][0]["mediaURL"]
            else:
                raise ValueError("The format is invalid: {}".format(content["entities"]["media"][0]))
        elif not keep_no_img_tweet:
            continue

        # If this tweet is a re-tweet from others, we should use the original location and time.
        if "retweeted_status" in content:
            if "location" in content["retweeted_status"]["user"]:
                key2val["location"] = content["retweeted_status"]["user"]["location"]
            else:
                key2val["location"] = ""
                location_missing_count += 1
            if "created_at" in content["retweeted_status"]:
                key2val["created_at"] = content["retweeted_status"]["created_at"]
            elif "createdAt" in content["retweeted_status"]:
                key2val["created_at"] = content["retweeted_status"]["createdAt"]
            else:
                raise ValueError("The format is invalid: {}".format(content["retweeted_status"]))
        else:
            if "location" in content["user"]:
                key2val["location"] = content["user"]["location"]
            else:
                key2val["location"] = ""
                location_missing_count += 1
            if "created_at" in content:
                key2val["created_at"] = content["created_at"]
            elif "createdAt" in content:
                key2val["created_at"] = content["createdAt"]
            else:
                raise ValueError("The format is invalid: {}".format(content))

        lat, lng = get_coordinate(location2latlng, key2val["location"], key2val["full_text"])
        # In current setting we just omit all tweets that cannot be labeled on the map.
        if lat is None and not keep_no_latlng_tweet:
            continue
        key2val["latitude"] = lat
        key2val["longitude"] = lng
        result.append(key2val)
    print("There are {} tweets don't have location info".format(location_missing_count))
    print("The final tweets number for display is {}".format(len(result)))
    return result


def write_display_info(display_info, outfile):
    """
    Write the display info to a file, and currently we use the js file which could be loaded directly by JavaScript.
    TODO(junpeiz): Write the data into a json file and import it to database.

    :param display_info: A list of dict containing the information needed to display.
    :param outfile: The path to the output file.

    :return: None
    """
    with open(outfile, 'w', encoding='utf8') as fout:
        fout.write("const tweets = ")
        fout.write(json.dumps(display_info, sort_keys=True, indent=2))
        fout.write(";\n")
        fout.write("\nexport default tweets;\n")


if __name__ == '__main__':
    tweetid_list, tweet_content_list = get_tweetid_content(["data/all-tweets-2019.txt"])
    display_info = get_info_for_display(tweet_content_list)
    write_display_info(display_info, "out/posts_full.js")
    # Filter out those tweets that has been predicted high scores by our model
    predict_output = os.path.join('eval', 'rf.run')
    threshold_score = 0.5
    valid_tweetid = set()
    with open(predict_output, 'r', encoding='utf8') as f:
        for line in f:
            line = line.strip().split('\t')
            tweetid = line[2]
            score = float(line[4])
            if score >= threshold_score:
                valid_tweetid.add(tweetid)
    filtered_tweet_content_list = [content for idx, content in enumerate(tweet_content_list)
                                   if tweetid_list[idx] in valid_tweetid]
    display_info = get_info_for_display(filtered_tweet_content_list, keep_no_img_tweet=False, keep_no_latlng_tweet=False)
    write_display_info(display_info, "out/posts_filtered.js")
