import jieba, os
from gensim import corpora, models
import re


def stopwordslist():
    """
    创建停用词列表
    """
    stopwords = [line.strip() for line in open('./stopwords.txt', encoding='UTF-8').readlines()]
    return stopwords


def seg_depart(sentence):
    """
    对句子进行中文分词
    """
    sentence_depart = jieba.cut(sentence.strip())
    stopwords = stopwordslist()
    outstr = ''
    for word in sentence_depart:
        if word not in stopwords:
            outstr += word
            outstr += " "
    return outstr


"""如果文档还没分词，就进行分词"""
if not os.path.exists('output/data_jieba.txt'):
    # 给出文档路径
    filename = "output/data.txt"
    outfilename = "output/data_jieba.txt"
    inputs = open(filename, 'r', encoding='UTF-8')
    outputs = open(outfilename, 'w', encoding='UTF-8')

    # 把非汉字的字符全部去掉
    # 将输出结果写入ouputs.txt中
    for line in inputs:
        print(line)
        # line = line.split('\t')[1]
        line = re.sub(r'[^\u4e00-\u9fa5]+', '', line)
        line_seg = seg_depart(line.strip())
        outputs.write(line_seg.strip() + '\n')

    outputs.close()
    inputs.close()
    print("删除停用词和分词成功！！！")

"""准备好训练语料，整理成gensim需要的输入格式"""
fr = open('output/data_jieba.txt', 'r', encoding='utf-8')
train = []
for line in fr.readlines():
    line = [word.strip() for word in line.split(' ')]
    train.append(line)

"""构建词频矩阵，训练LDA模型"""
dictionary = corpora.Dictionary(train)
# corpus[0]: [(0, 1), (1, 1), (2, 1), (3, 1), (4, 1),...]
# corpus是把每条专利ID化后的结果，每个元素是专利中的每个词语，在字典中的ID和频率
corpus = [dictionary.doc2bow(text) for text in train]

print("词典中单词个数：", len(dictionary))
lda = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=5, )
# k越多，训练越复杂，但是，如果数据量不多的情况下，很难提取出小样本数据的主题
# 所以主题挖掘的应用到底在哪？
topic_list = lda.print_topics(8)

print("\n5个主题的单词分布为：\n")
for topic in topic_list:
    print(topic)
lda.save("model/lda.model")
"""抽取专利的主题"""


def Topic(file_test="output/test.txt"):
    # 用来测试的专利
    news_test = open(file_test, 'r', encoding='UTF-8')

    test = []
    count = 0
    # 处理成正确的输入格式
    for line in news_test:
        # line = line.split('\t')[1]
        line = re.sub(r'[^\u4e00-\u9fa5]+', '', line)
        line_seg = seg_depart(line.strip())
        line_seg = [word.strip() for word in line_seg.split(' ')]
        test.append(line_seg)
        count += 1

    # 专利ID化
    corpus_test = [dictionary.doc2bow(text) for text in test]
    # 得到每条专利的主题分布
    topics_test = lda.get_document_topics(corpus_test)

    for i in range(count):
        print(str(i + 1) + ":", topics_test[i], '\n')
    fr.close()
    news_test.close()


Topic()
