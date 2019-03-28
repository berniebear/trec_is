#!/usr/bin/env bash

java -jar TREC-IS-Client-v2.jar --noNewLinesInTweet trecis2018-train

java -jar TREC-IS-Client-v2.jar --noNewLinesInTweet trecis2018-test
