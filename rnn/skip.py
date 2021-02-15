
import nltk
from nltk.corpus import brown
from gensim.models import Word2Vec
import multiprocessing
import datetime

def logmsg(vmsg):
  tstamp = "["+str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))+"]"
  print("{} {}".format(tstamp, vmsg), flush=True)

logmsg('getting sentences')
sentences = brown.sents

logmsg('importing downloader')
import gensim.downloader
# Show all available models in gensim-data
logmsg('models:')
for key in list(gensim.downloader.info()['models'].keys()):
    logmsg(f'\t{key}')
# Download the "glove-twitter-25" embeddings
logmsg('load')
word_vectors = gensim.downloader.load('glove-twitter-25')
#glove_vectors = gensim.downloader.load('glove-wiki-gigaword-300')

logmsg('lookup')
# Use the downloaded vectors as usual:
logmsg(f"twitter: {word_vectors.most_similar('twitter')}")

#nltk.download()




