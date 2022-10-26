import os, re, sys

from optparse import OptionParser

import datetime, pytz, json
from dateutil.tz import tzlocal

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.patches as patches
import matplotlib.path as path
import matplotlib.cm as cm

from collections import Counter

from nltk.cluster import GAAClusterer
import nltk.corpus
from nltk import decorators
import nltk.stem

nltk.download("stopwords")

stemmer_func = nltk.stem.snowball.EnglishStemmer().stem
stopwords = set(nltk.corpus.stopwords.words("english"))

DEBUG = False


def iterkeys(d):
    """Python 2/3 compatibility function for dict.iterkeys()"""
    if sys.version_info >= (3,):
        return d.keys()
    else:
        return d.iterkeys()


def load_tweets_from_js(js_file):
    with open(js_file, "r") as f:
        data = f.read()
        data = data.replace("window.YTD.tweet.part0 = ", "")
        tweets = json.loads(data)
        for tweet in tweets:
            ts = datetime.datetime.strptime(
                tweet["tweet"]["created_at"], "%a %b %d %H:%M:%S +0000 %Y"
            )
            ts = ts.replace(tzinfo=pytz.utc)
            ts = ts.astimezone(tzlocal())
            tweet["timestamp"] = ts
        print("Loaded %d tweets" % len(tweets))
        return tweets


def by_hour(tweets, out_dir):
    hours = []
    for tweet in tweets:
        hours.append(tweet["timestamp"].hour)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    n, bins = np.histogram(hours, range(25))

    print(n, bins)

    # get the corners of the rectangles for the histogram
    left = np.array(bins[:-1])
    right = np.array(bins[1:])
    bottom = np.zeros(len(left))
    top = bottom + n

    # we need a (numrects x numsides x 2) numpy array for the path helper
    # function to build a compound path
    XY = np.array([[left, left, right, right], [bottom, top, top, bottom]]).T

    # get the Path object
    barpath = path.Path.make_compound_path_from_polys(XY)

    # make a patch out of it
    patch = patches.PathPatch(barpath, facecolor="blue", edgecolor="gray", alpha=0.8)
    ax.add_patch(patch)

    # update the view limits
    ax.set_xlim(left[0], right[-1])
    ax.set_ylim(bottom.min(), top.max())

    # plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.xticks(range(0, 24), ha="center")

    plt.xlabel("Hour")
    plt.ylabel("# Tweets")
    plt.title("# of Tweets by Hour")

    plt.savefig(os.path.join(out_dir, "by-hour.png"), bbox_inches=0)
    if DEBUG:
        plt.show()


def by_dow(tweets, out_dir):
    dow = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
    c = Counter()
    for tweet in tweets:
        c[tweet["timestamp"].strftime("%A")] += 1
    print(c.most_common(10))

    N = len(dow)

    ind = np.arange(N)
    width = 0.9

    fig = plt.figure()
    ax = fig.add_subplot(111)

    rects1 = ax.bar(0.05 + ind, [c[d] for d in dow], width, color="b")

    ax.set_ylabel("# Tweets")
    ax.set_title("Tweets by Day of Week")
    ax.set_xticks(ind + 0.5 * width)
    ax.set_xticklabels([d[:3] for d in dow])

    plt.savefig(os.path.join(out_dir, "by-dow.png"), bbox_inches=0)
    if DEBUG:
        plt.show()


def by_month(tweets, out_dir):
    c = Counter()
    for tweet in tweets:
        c[tweet["timestamp"].strftime("%Y-%m")] += 1
    print(c.most_common(10))

    N = len(c)

    ind = np.arange(N)  # the x locations for the groups
    width = 0.8  # the width of the bars

    fig = plt.figure()
    ax = fig.add_subplot(111)

    rects1 = ax.bar(ind, [c[x] for x in sorted(c.keys())], width, color="b")

    ax.set_ylabel("# Tweets")
    ax.set_title("Tweets by Month")

    ax.set_xticks([i for i, x in enumerate(sorted(c.keys())) if i % 6 == 0])
    ax.set_xticklabels(
        [x for i, x in enumerate(sorted(c.keys())) if i % 6 == 0], rotation=30
    )

    plt.savefig(os.path.join(out_dir, "by-month.png"), bbox_inches=0)
    if DEBUG:
        plt.show()


def by_month_dow(tweets, out_dir):
    dow = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
    # Get the # of week and weekday for each tweet
    data = {}
    for tweet in tweets:
        weekday = tweet["timestamp"].strftime("%A")
        iso_yr, iso_wk, iso_wkday = tweet["timestamp"].isocalendar()
        # key = str(iso_yr) + '-' + str(iso_wk)
        key = tweet["timestamp"].strftime("%Y-%m")
        if key not in data:
            data[key] = Counter()
        data[key][weekday] += 1
    print(data)
    # Convert to numpy
    xs = []
    ys = []
    a = np.zeros((7, len(data)))
    for i, key in enumerate(sorted(iterkeys(data))):
        for j, d in enumerate(dow):
            # a[j,i] = data[key][d]
            for k in range(data[key][d]):
                xs.append(j)
                ys.append(i)
    # Convert to x,y pairs
    # heatmap, xedges, yedges = np.histogram2d(np.array(xs), np.array(ys), bins=(7,len(data)))
    # extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    # plt.clf()
    # plt.imshow(heatmap, extent=extent)
    # if DEBUG:
    #     plt.show()

    x = np.array(xs)
    y = np.array(ys)

    gridsize = 30
    plt.hexbin(x, y, C=None, gridsize=gridsize, cmap=cm.jet, bins=None)
    plt.axis([x.min(), x.max(), y.min(), y.max()])

    plt.title("Tweets by Day of Week and Month")
    plt.xlabel("Day of Week")
    plt.ylabel("Month")
    plt.gca().set_xticklabels([d[:3] for d in dow])
    plt.gca().set_yticklabels(
        [key for i, key in enumerate(sorted(iterkeys(data))) if i % 6 == 0]
    )
    plt.gca().set_yticks(
        [i for i, key in enumerate(sorted(iterkeys(data))) if i % 6 == 0]
    )

    print(list(sorted(iterkeys(data))))

    cb = plt.colorbar()
    cb.set_label("# Tweets")

    plt.savefig(os.path.join(out_dir, "by-month-dow.png"), bbox_inches=0)
    if DEBUG:
        plt.show()


def by_month_length(tweets, out_dir):
    c = Counter()
    s = Counter()
    for tweet in tweets:
        c[tweet["timestamp"].strftime("%Y-%m")] += 1
        s[tweet["timestamp"].strftime("%Y-%m")] += len(tweet["tweet"]["full_text"])
    print(c.most_common(10))

    N = len(c)
    ind = np.arange(N)
    width = 0.8

    fig = plt.figure()
    ax = fig.add_subplot(111)

    rects1 = ax.bar(ind, [s[x] / c[x] for x in sorted(c.keys())], width, color="b")

    ax.set_ylabel("Avg Tweet Length")
    ax.set_title("Avg Tweet Length by Month")

    ax.set_xticks([i for i, x in enumerate(sorted(c.keys())) if i % 6 == 0])
    ax.set_xticklabels(
        [x for i, x in enumerate(sorted(c.keys())) if i % 6 == 0], rotation=30
    )

    plt.savefig(os.path.join(out_dir, "by-month-length.png"), bbox_inches=0)
    if DEBUG:
        plt.show()


def by_month_type(tweets, out_dir):
    c_total = Counter()
    c_tweets = Counter()
    c_rts = Counter()
    c_replies = Counter()
    months = set()
    for tweet in tweets:
        key = tweet["timestamp"].strftime("%Y-%m")
        months.add(key)
        c_total[key] += 1
        if "in_reply_to_status_id" in tweet["tweet"]:
            c_replies[key] += 1
        elif tweet["tweet"]["retweeted"]:
            c_rts[key] += 1
        else:
            c_tweets[key] += 1

    months = list(sorted(months))
    N = len(months)
    ind = np.arange(N)

    # Create the non stacked version
    width = 0.3

    fig = plt.figure()
    ax = fig.add_subplot(111)

    rects1 = ax.bar(ind, [c_tweets[m] for m in months], width, color="r")
    rects2 = ax.bar(ind + width, [c_rts[m] for m in months], width, color="b")
    rects3 = ax.bar(ind + width * 2, [c_replies[m] for m in months], width, color="g")

    ax.set_ylabel("# Tweets")
    ax.set_title("Type of Tweet by Month")

    ax.set_xticks([i + width for i, x in enumerate(months) if i % 6 == 0])
    ax.set_xticklabels([x for i, x in enumerate(months) if i % 6 == 0], rotation=30)

    ax.legend((rects1[0], rects2[0], rects3[0]), ("Tweet", "RT", "Reply"))

    fig.set_size_inches(12, 6)
    plt.savefig(os.path.join(out_dir, "by-month-type.png"), bbox_inches=0)
    if DEBUG:
        plt.show()

    # Create the stacked version
    width = 0.9

    fig = plt.figure()
    ax = fig.add_subplot(111)

    d_tweets = np.array([float(c_tweets[m]) / c_total[m] for m in months])
    d_rts = np.array([float(c_rts[m]) / c_total[m] for m in months])
    d_replies = np.array([float(c_replies[m]) / c_total[m] for m in months])

    rects1 = ax.bar(ind + width / 2, d_tweets, width, color="r")
    rects2 = ax.bar(ind + width / 2, d_rts, width, bottom=d_tweets, color="b")
    rects3 = ax.bar(
        ind + width / 2, d_replies, width, bottom=d_tweets + d_rts, color="g"
    )

    ax.set_ylabel("Tweet Type %")
    ax.set_title("Type of Tweet by Month")

    ax.set_xticks([i for i, x in enumerate(months) if i % 6 == 0])
    ax.set_xticklabels([x for i, x in enumerate(months) if i % 6 == 0], rotation=30)

    ax.legend((rects1[0], rects2[0], rects3[0]), ("Tweet", "RT", "Reply"), loc=4)

    plt.savefig(os.path.join(out_dir, "by-month-type-stacked.png"), bbox_inches=0)
    if DEBUG:
        plt.show()


@decorators.memoize
def get_words(tweet_text):
    return [word.lower() for word in re.findall("\w+", tweet_text) if len(word) > 3]


def word_frequency(tweets, out_dir):
    c = Counter()
    hash_c = Counter()
    at_c = Counter()
    for tweet in tweets:
        for word in get_words(tweet["tweet"]["full_text"]):
            c[word] += 1
        for word in re.findall("@\w+", tweet["tweet"]["full_text"]):
            at_c[word.lower()] += 1
        for word in re.findall("\#[\d\w]+", tweet["tweet"]["full_text"]):
            hash_c[word.lower()] += 1
    print(c.most_common(50))
    print(hash_c.most_common(50))
    print(at_c.most_common(50))


@decorators.memoize
def normalize_word(word):
    return stemmer_func(word.lower())


@decorators.memoize
def vectorspaced(tweet_text, all_words):
    components = [normalize_word(word) for word in get_words(tweet_text)]
    return np.array(
        [word in components and word not in stopwords for word in all_words], np.short
    )


# TODO: Fix this
def get_word_clusters(tweets):
    all_words = set()
    for tweet in tweets:
        for word in get_words(tweet["tweet"]["full_text"]):
            all_words.add(word)
    all_words = tuple(all_words)

    cluster = GAAClusterer(5)
    cluster.cluster(
        [vectorspaced(tweet["tweet"]["full_text"], all_words) for tweet in tweets]
    )

    classified_examples = [
        cluster.classify(vectorspaced(tweet["tweet"]["full_text"], all_words))
        for tweet in tweets
    ]

    for cluster_id, title in sorted(zip(classified_examples, job_titles)):
        print(cluster_id, title)


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option(
        "-j",
        "--js",
        dest="js_file",
        help="Twitter archive JavaScript file",
        metavar="FILE",
    )
    parser.add_option(
        "-o", "--out", dest="out_directory", help="Output directory", metavar="FILE"
    )

    (options, args) = parser.parse_args()

    if options.js_file is None:
        print("You must pass a JavaScript archive")
        exit(1)

    output_dir = options.out_directory
    if output_dir is None:
        output_dir = "out"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    all_tweets = load_tweets_from_js(options.js_file)

    by_month(all_tweets, output_dir)
    by_month_type(all_tweets, output_dir)
    by_month_length(all_tweets, output_dir)
    by_month_dow(all_tweets, output_dir)
    by_dow(all_tweets, output_dir)
    by_hour(all_tweets, output_dir)
    word_frequency(all_tweets, output_dir)
    # get_word_clusters(all_tweets, output_dir)
