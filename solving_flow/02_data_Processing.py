import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

file_path = 'F:\\Competition_2025\\BDAIAC\\AIsoft_2022\\data\\task2\\task2_1.txt'
with open(file_path, 'r', encoding='utf-8') as f:
    documents = [line.strip().lower() for line in f.readlines()]

# 使用sklearn计算TF-IDF
'''
TF-IDF介绍：
TF-IDF（Term Frequency-Inverse Document Frequency）是一种用于评估文本中词语重要性的方法。
TF-IDF是由两个部分组成：
1. 词频（Term Frequency）：TF表示词语在文档中出现的频率。
2. 逆文档频率（Inverse Document Frequency）：IDF表示词语在整个文档集中的逆频率。
TF-IDF的计算公式如下：
TF-IDF = TF * IDF
其中，TF表示词语在文档中出现的频率，IDF表示词语在整个文档集中的逆频率。
TF-IDF的计算方法如下：
1. 计算词频：TF = 词语在文档中出现的次数 / 文档中词语的总数
2. 计算逆文档频率：IDF = log(文档集中的文档总数 / 包含词语的文档数 + 1)
3. 计算TF-IDF：TF-IDF = TF * IDF
'''

vectorizer = TfidfVectorizer()  # 初始化TF-IDF向量器
tfidf_matrix = vectorizer.fit_transform(documents)  # 计算TF-IDF矩阵

feature_names = vectorizer.get_feature_names_out()  # 获取特征名称

results = []

for i, doc in enumerate(documents):
    doc_tfidf = tfidf_matrix[i]  # 获取第i个文档的TF-IDF向量
    word_tfidf = [(feature_names[j], doc_tfidf[0, j])
                  for j in range(len(feature_names)) if doc_tfidf[0, j] > 0]

    # 按TF-IDF值降序排序
    word_tfidf.sort(key=lambda x: x[1], reverse=True)

    results.append(f"语句{i + 1}：{word_tfidf}")

for result in results:
    print(result)

# 写入result_task2_1文件
output_file_path = 'F:\\Competition_2025\\BDAIAC\\AIsoft_2022\\result\\result_task2_1.txt'
with open(output_file_path, 'w', encoding = 'utf-8') as f:
    for result in results:
        f.write(result + '\n')
print("结果已写入文件")