from django.shortcuts import render
from sklearn.cross_validation import train_test_split
import numpy as np
import gensim

# Create your views here.

from .models import HistoryInfo

def add_view_log(request):  #  调用接口记录访问日志
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR', '')
    if x_forwarded_for:
        ip = x_forwarded_for.split(',')[0]  # 这里是真实的ip
    else:
        ip = request.META.get('REMOTE_ADDR')  # 这里获得代理ip
        HistoryInfo.objects.create(hostname=ip)

    return render(request, 'index.html', {
        'hostname': ip,
        'count': HistoryInfo.objects.count()
    })

def get_all_view_logs(request):  # 获取所有的访问日志的记录，返回json数据格式
    result = HistoryInfo.objects.all()
    json_data = HistoryInfo.serialize("json", result, ensure_ascii=False)
    return HistoryInfo(json_data, content_type='application/json; charset=utf-8')



LabeledSentence = gensim.models.doc2vec.LabeledSentence



with open('IMDB_data/pos.txt', 'r') as infile:
    pos_reviews = infile.readlines()

with open('IMDB_data/neg.txt', 'r') as infile:
    neg_reviews = infile.readlines()

with open('IMDB_data/unsup.txt', 'r') as infile:
    unsup_reviews = infile.readlines()

# 1 代表积极情绪，0 代表消极情绪
y = np.concatenate((np.ones(len(pos_reviews)), np.zeros(len(neg_reviews))))

x_train, x_test, y_train, y_test = train_test_split(np.concatenate((pos_reviews, neg_reviews)), y, test_size=0.2)


# 零星的预处理
def cleanText(corpus):
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


# Gensim 的 Doc2Vec 工具要求每个文档/段落包含一个与之关联的标签。我们利用 LabeledSentence 进行处理。格式形如 “TRAIN_i” 或者 “TEST_i”，其中 “i” 是假的评论索引。
def labelizeReviews(reviews, label_type):
    labelized = []
    for i, v in enumerate(reviews):
        label = '%s_%s' % (label_type, i)
        labelized.append(LabeledSentence(v, [label]))
    return labelized


x_train = labelizeReviews(x_train, 'TRAIN')
x_test = labelizeReviews(x_test, 'TEST')
unsup_reviews = labelizeReviews(unsup_reviews, 'UNSUP')
