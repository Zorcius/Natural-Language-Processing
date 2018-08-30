from gensim.models import word2vec
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

#sentences = word2vec.Text8Corpus("C:\\Zorcius\\filtering\\neg_seg.txt")
sentences = open("C:\\Zorcius\\filtering\\neg-seg.txt").readlines()

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(sentences)
#print(X)

#查看词频
from bokeh.models import ColumnDataSource, LabelSet
from bokeh.plotting import figure, show, output_file
'''
hist, edges = np.histogram(count_top1000,density=True, bins=100,normed=True)
p = figure(tools='pan,wheel_zoom,reset,save',
           toolbar_location='above',
           title='Top-1000 words distribution')
p.quad(top=hist,bottom=0,left=edges[:-1],right=edges[1:],line_color='#555555')
show(p)
'''
#Train
from gensim import corpora
model = word2vec.Word2Vec(sentences, size=200,window=5,min_count=10,workers=4)


