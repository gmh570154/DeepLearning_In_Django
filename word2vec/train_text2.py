#-*- coding: utf-8 -*-
from sklearn.model_selection import train_test_split
from gensim.models.word2vec import Word2Vec
import numpy as np
import os
import gensim
import random
from sklearn.linear_model import SGDClassifier

LabeledSentence = gensim.models.doc2vec.LabeledSentence
path = os.getcwd()
size = 400

with open(path[:-9] + '\\aclImdb\\train\\pos_all.txt', 'r') as infile:
    pos_tweets = infile.readlines()

with open(path[:-9] + '\\aclImdb\\train\\neg_all.txt', 'r') as infile:
    neg_tweets = infile.readlines()

with open(path[:-9] + '\\aclImdb\\train\\unsup_all.txt', 'r') as infile:
    unsup_reviews = infile.readlines()

# 1 代表积极情绪，0 代表消极情绪
y = np.concatenate((np.ones(len(pos_tweets)), np.zeros(len(neg_tweets))))

x_train, x_test, y_train, y_test = train_test_split(np.concatenate((pos_tweets, neg_tweets)), y, test_size=0.2)


def cleanText(corpus):  # 零星的预处理
    punctuation = """.,?!:;(){}[]"""
    corpus = [z.lower().replace('n', '') for z in corpus]
    corpus = [z.replace('&lt;br /&gt;', ' ') for z in corpus]
    # 将标点视为一个单词
    for c in punctuation:
        corpus = [z.replace(c, ' %s ' % c) for z in corpus]
    corpus = [z.split() for z in corpus]
    return corpus


x_train = cleanText(x_train)
x_test = cleanText(x_test)
unsup_reviews = cleanText(unsup_reviews)


# n_dim = 100
# # 初始化模型并创建词汇表（vocab）
# imdb_w2v = Word2Vec(size=n_dim, window=5, min_count=1, workers=12)
# imdb_w2v.build_vocab(x_train)
# # 训练模型 (会花费几分钟)
# imdb_w2v.train(x_train, total_examples=imdb_w2v.corpus_count, epochs=imdb_w2v.iter)
# current_dir = os.path.dirname(__file__)
# imdb_w2v.save("train_x.model")
# result = imdb_w2v.most_similar(positive=['biggest','small'], negative=['big'], topn=5)
# print(result)
# exit(0)

# Gensim 的 Doc2Vec 工具要求每个文档/段落包含一个与之关联的标签。我们利用 LabeledSentence 进行处理。
# 格式形如 “TRAIN_i” 或者 “TEST_i”，其中 “i” 是传的评论索引。
def labelizeReviews(reviews, label_type):
    labelized = []
    for i, v in enumerate(reviews):  # 返回索引及每一项的值
        label = '%s_%s' % (label_type, i)
        labelized.append(LabeledSentence(v, [label]))
    return labelized


x_train_tag = labelizeReviews(x_train, 'TRAIN')
x_test_tag = labelizeReviews(x_test, 'TEST')
unsup_reviews_tag = labelizeReviews(unsup_reviews, 'UNSUP')


# 实例化 DM 和 DBOW 模型
model_dm = gensim.models.Doc2Vec(min_count=1, window=10, vector_size=size, sample=1e-3, negative=5, workers=3)
model_dbow = gensim.models.Doc2Vec(min_count=1, window=10, vector_size=size, sample=1e-3, negative=5, dm=0, workers=3)

#alldata = np.concatenate((x_train_tag, x_test_tag, unsup_reviews_tag))
# arr2 = np.concatenate((x_train, x_test, unsup_reviews))
import copy
alldata = []
alldata.extend(x_train_tag)
alldata.extend(x_test_tag)
alldata.extend(unsup_reviews_tag)


# 对所有评论创建词汇表，使用所有的数据建立词典
model_dm.build_vocab(alldata)
model_dbow.build_vocab(alldata)
alldata2 = []
alldata2.extend(x_train_tag)
alldata2.extend(unsup_reviews_tag)
alldata3 = np.concatenate((alldata2))


def sentences_perm(sentences):
    shuffled = list(sentences)
    random.shuffle(shuffled)
    return (shuffled)


for epoch in range(1):
    model_dm.train(sentences_perm(alldata2), total_examples=model_dm.corpus_count,epochs=1)
    model_dbow.train(sentences_perm(alldata2), total_examples=model_dbow.corpus_count,epochs=1)


# 从我们的模型中获得训练过的向量
def getVecs(model, corpus, size):
    vecs = [np.array(model[z.tags[0]]).reshape((1, size)) for z in corpus]
    return np.concatenate(vecs)  # 按轴axis连接array组成一个新的array

train_vecs_dm = getVecs(model_dm, x_train_tag, size)
train_vecs_dbow = getVecs(model_dbow, x_train_tag, size)
train_vecs = np.hstack((train_vecs_dm, train_vecs_dbow))  # 在水平方向上平铺

# 训练测试数据集
for epoch in range(1):
    model_dm.train(sentences_perm(x_train_tag), total_examples=model_dm.corpus_count, epochs=1)
    model_dbow.train(sentences_perm(x_train_tag), total_examples=model_dbow.corpus_count, epochs=1)

model_dbow.save("dbow.model")
model_dm.save("db.model")

# 创建测试数据集向量
test_vecs_dm = getVecs(model_dm, x_train_tag, size)
test_vecs_dbow = getVecs(model_dbow, x_train_tag, size)
test_vecs = np.hstack((test_vecs_dm, test_vecs_dbow))
lr = SGDClassifier(loss='log', penalty='l1')
lr.fit(train_vecs, y_train)
#lr_y_predict = lr.predict(y_test.shape[0])
print('Test Accuracy: %.2f' % lr.score(test_vecs, y_test))
