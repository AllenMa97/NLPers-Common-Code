from transformers import BertTokenizer, BertModel

# BERT隐层大小：768
tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
model = BertModel.from_pretrained("hfl/chinese-roberta-wwm-ext")

############  单句子案例  ############
print("单句子案例")
# 批数=1， 词数=165
my_text = '它考虑到文本、语音、图像、视频等数据的上下文环境，以及数据之间的关系和上下文信息的影响。在这种方法中，学习算法会利用上下文信息来提高预测和分类的准确性和有效性。例如，在自然语言处理中，上下文学习可以帮助机器学习算法更好地理解一个句子中的词语含义和关系。在计算机视觉中，它可以帮助机器学习算法更好地识别图像中不同物体之间的关系。'

# token_encoding
token_encoding = tokenizer(text=my_text, return_tensors='pt')
print(token_encoding)

# embedding尺寸：【批数，词数，768】
embedding = model(**token_encoding).last_hidden_state
print(embedding)


############  句子列表案例  ############
print("句子列表案例")
# 批数=3，批内最大词数=85
my_text_list = [
    '它考虑到文本、语音、图像、视频等数据的上下文环境，以及数据之间的关系和上下文信息的影响。',
    '在这种方法中，学习算法会利用上下文信息来提高预测和分类的准确性和有效性。例如，在自然语言处理中，上下文学习可以帮助机器学习算法更好地理解一个句子中的词语含义和关系。',
    '在计算机视觉中，它可以帮助机器学习算法更好地识别图像中不同物体之间的关系。'
]

token_encoding_2 = tokenizer(text=my_text_list, return_tensors='pt', padding=True, truncation=True)
print(token_encoding_2)

# embedding尺寸：【批数，批内最大词数，768】
embedding_2 = model(**token_encoding_2).last_hidden_state
print(embedding_2)


print("Finish！")

