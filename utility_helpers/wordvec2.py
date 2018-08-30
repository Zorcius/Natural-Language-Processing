import numpy as np
import os
from random import shuffle
import re
from bokeh.models import ColumnDataSource, LabelSet
from bokeh.plotting import figure, show, output_file

import urllib.request
import zipfile
import lxml.etree

with zipfile.ZipFile('C:\\Zorcius\\filtering\\wikitext-103-raw-v1.zip', 'r') as z:
    input_text = str(z.open('wikitext-103-raw/wiki.train.raw', 'r').read(), encoding='utf-8')
sentences_wiki = []
for line in input_text.split('\n'):
    s = [x for x in line.split('.') if x and len(x.split()) >= 5]
    sentences_wiki.extend(s)
for s_i in range(len(sentences_wiki)):
    sentences_wiki[s_i] = re.sub("[^a-z]"," ", sentences_wiki[s_i].lower())
    sentences_wiki[s_i] = re.sub(r'\([^)]*\)','',sentences_wiki[s_i])
del input_text

#sample 1/5 of the data
shuffle(sentences_wiki)
sentences_wiki = sentences_wiki[:int(len(sentences_wiki)/5)]

#分词
sentences_strings_wiki = sentences_wiki
sentences_wiki = []
for sent_str in sentences_strings_wiki:
    tokens = re.sub(r"[^a-z0-9]+",' ',sent_str.lower()).split()
    sentences_wiki.append(tokens)
from gensim.models import word2vec
from gensim import corpora
model_wiki = word2vec.Word2Vec(sentences_wiki, size=100, window=5,min_count=10,workers=4)
#建立词典
dictionary_wiki = corpora.Dictionary(sentences_wiki)
print(len(dictionary_wiki))
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(sentences_strings_wiki)  #这里CountVectorizer会自动对语句进行分词
words_top_wiki_index = np.array(np.argsort(-X.sum(axis = 0))).squeeze()[:1000]
words_top_wiki_word = np.array(vectorizer.get_feature_names())[words_top_wiki_index]

words_top_wiki = words_top_wiki_word
#tsne visualization
# This assumes words_top_wiki is a list of strings, the top 1000 words
words_top_vec_wiki = model_wiki[words_top_wiki]
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, random_state=0)
words_top_wiki_tsne = tsne.fit_transform(words_top_vec_wiki)
p = figure(tools="pan,wheel_zoom,reset,save",
           toolbar_location="above",
           title="word2vec T-SNE for most common words")

source = ColumnDataSource(data=dict(x1=words_top_wiki_tsne[:,0],  #横坐标
                                    x2=words_top_wiki_tsne[:,1],  #纵坐标
                                    names=words_top_wiki))        #还可以贴标签

p.scatter(x="x1", y="x2", size=8, source=source)

labels = LabelSet(x="x1", y="x2", text="names", y_offset=6,
                  text_font_size="8pt", text_color="#555555",
                  source=source, text_align='center')
p.add_layout(labels)

show(p)

from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D

estimator = KMeans(n_clusters=100)

estimator.fit(words_top_vec_wiki)
est_labels = estimator.labels_

print (words_top_wiki_word[est_labels == 0])
print (words_top_wiki_word[est_labels == 1])
colors = [
    "#%02x%02x%02x" % (int(2.55 * r), 150, 150) for r in est_labels
]

cluster_source = ColumnDataSource(data=dict(x1=words_top_wiki_tsne[:,0],  #横坐标
                                            x2=words_top_wiki_tsne[:,1],          #纵坐标
                                            names=words_top_wiki))        #还可以贴标签

p = figure(tools="pan,wheel_zoom,reset,save",
           toolbar_location="above",
           title="word2vec T-SNE for most common words")

p.scatter(x="x1", y="x2", size=8, source=cluster_source,
          fill_color=colors, #加上颜色
          fill_alpha=0.6,
          #radius=radii,
          line_color=None)

labels = LabelSet(x="x1", y="x2", text="names", y_offset=6,
                  text_font_size="8pt", text_color="#555555",
                  source=source, text_align='center')

p.add_layout(labels)

show(p)  # open a browser