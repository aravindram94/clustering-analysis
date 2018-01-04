import glob
import operator
import re
import math

import time

from porter_stemmer import getStemWord;

articles_count = {}

data_set = [];

def getTopicName(line):
    line = line[11:]
    line = line[:-14]
    return line


start_time = time.time()

# loading the stoplist
stoplist = [];
lines = [line.rstrip('\n') for line in open('./stoplist.txt')];

for line in lines:
    for word in line.split():
        stoplist.append(word);

newid = "";
body_text = [];
topic = "";
body_flag = 0;

for filename in glob.glob('./reuters21578/*.sgm'):
    with open(filename) as f:
        for line in f:

            if line.startswith("<REUTERS"):
                newid = "";
                body_text = [];
                topic = "";
                if line.find("NEWID=") != -1:
                    start_index = line.find("NEWID=") + 7;
                    newid = line[start_index:-3];

            if line.find("</BODY>") != -1:
                body_text.pop();
                body_flag = 0;

            if body_flag == 1:
                body_text.append(line);

            if line.find("<BODY>") != -1:
                start_index = line.find("<BODY>") + 6;
                body_text.append(line[start_index:])
                body_flag = 1;

            if line.startswith("</REUTERS>"):
                if topic != "" and len(body_text) != 0:
                    if topic in articles_count:
                        articles_count[topic] = articles_count[topic] + 1;
                    else:
                        articles_count[topic] = 1;

                    data_set.append({"new_id": newid, "body_text": body_text, "topic": topic});
                    newid = "";
                    body_text = [];
                    topic = "";

            if line.startswith("<TOPICS><D>") and len(line.split("<D>")) == 2:
                topic = getTopicName(line);

# filtering the top 20 topics
sorted_article_count = sorted(articles_count.items(), key=operator.itemgetter(1), reverse=True)
count = 0
filtered_topics = [];
for article in sorted_article_count[:20]:
    count += article[1]
    filtered_topics.append(article[0]);

print "Frequent topics filtering done. Total articles count : ", count

overall_tokens_count = {};

def cleanBodyText(body_text):
    result = [];
    for line in body_text:

        # remove non ascii
        line = ''.join([c for c in line if 0 < ord(c) < 127])

        # convert to lower
        line = line.lower();

        # replace non alpha with space
        line = re.sub("[^a-zA-Z0-9]+", " ", line);

        # split using whitespace
        tokens = line.split();

        for token in tokens:

            # remove tokens which are completely digits and remove all the stopwords
            if (not token.isdigit()) and (token not in stoplist):

                # get the stem word
                stem = getStemWord(token);

                # update the overall token count
                if stem in overall_tokens_count:
                    overall_tokens_count[stem] = overall_tokens_count[stem] + 1;
                else:
                    overall_tokens_count[stem] = 1;

                # put the generated token in result
                result.append(stem);

    return result;

# data cleaning

filtered_data_set = [];

for data in data_set:
    if data["topic"] in filtered_topics:
        filtered_data_set.append({"topic": data["topic"], "new_id": data["new_id"], "tokens":cleanBodyText(data["body_text"])});

print "body text cleaning done . frequent data set size : "+ str(len(filtered_data_set));

# eliminating the infrequent tokens
frequent_tokens = [];
for token in overall_tokens_count:
    if overall_tokens_count[token] >= 5:
        frequent_tokens.append(token);

clabel_file = open("reuters21578.clabel", "w");
for index, token in enumerate(frequent_tokens):
    clabel_file.write(str(index)+","+str(token) + "\n");

clabel_file.close()

print "infrequent tokens elimination done, frequent tokens size : "+ str(len(frequent_tokens));

#vector models
frequency_based_vector_model = [];
sqrt_frequency_based_vector_model = [];
log_frequency_based_vector_model = [];

frequency_based_vector_model_file = open("freq.csv", "w");
sqrt_frequency_based_vector_model_file = open("sqrtfreq.csv", "w");
log_frequency_based_vector_model_file = open("log2freq.csv", "w");
class_file = open("reuters21578.clas", "w");

for data in filtered_data_set:

    class_file.write(str(data["new_id"]) + "," + str(data["topic"]) + "\n");

    counts = {token: data["tokens"].count(token) for token in data["tokens"]};

    # calculate sum for normalizing
    sum_freq = 0;
    sum_sqrt = 0;
    sum_log = 0;

    for index, token in enumerate(frequent_tokens):
        if token in data["tokens"]:
            sum_freq += math.pow(counts[token], 2);
            sum_sqrt += math.pow((1 + math.sqrt(counts[token])), 2);
            sum_log += math.pow((1 + math.log(counts[token], 2)), 2);

    sum_freq = math.sqrt(sum_freq);
    sum_sqrt = math.sqrt(sum_sqrt);
    sum_log = math.sqrt(sum_log);

    for index, token in enumerate(frequent_tokens):

        if token in data["tokens"]:
            norm_freq = (counts[token] / sum_freq);
            norm_sqrt = ((1 + math.sqrt(counts[token])) / sum_sqrt);
            norm_log = ((1 + math.log(counts[token], 2)) / sum_log);

            frequency_based_vector_model.append((data["new_id"], index, norm_freq));
            sqrt_frequency_based_vector_model.append((data["new_id"], index, norm_sqrt));
            log_frequency_based_vector_model.append((data["new_id"], index, norm_log));

            frequency_based_vector_model_file.write(str(data["new_id"]) + "," + str(index) + "," + str(norm_freq) + "\n");
            sqrt_frequency_based_vector_model_file.write(str(data["new_id"]) + "," + str(index) + "," + str(norm_sqrt) + "\n");
            log_frequency_based_vector_model_file.write(str(data["new_id"]) + "," + str(index) + "," + str(norm_log) + "\n");


frequency_based_vector_model_file.close()
sqrt_frequency_based_vector_model_file.close()
log_frequency_based_vector_model_file.close()

print "DONE"