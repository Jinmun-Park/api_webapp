import pandas as pd
import re
from sklearn.model_selection import train_test_split

import tensorflow as tf
import torch, gc

from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
import urllib.request

import numpy as np
import random
import time
import datetime

###############
# TORCH GPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print('No GPU available, using the CPU instead.')

model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=7)
model.cuda()
model.load_state_dict(torch.load('bert_model.pth'))
model.eval()

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
print("hi")

###############
def convert_input_data(sentences):
    # BERT의 토크나이저로 문장을 토큰으로 분리
    tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]

    # 입력 토큰의 최대 시퀀스 길이
    MAX_LEN = 128

    # 토큰을 숫자 인덱스로 변환
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]

    # 문장을 MAX_LEN 길이에 맞게 자르고, 모자란 부분을 패딩 0으로 채움
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

    # 어텐션 마스크 초기화
    attention_masks = []

    # 어텐션 마스크를 패딩이 아니면 1, 패딩이면 0으로 설정
    # 패딩 부분은 BERT 모델에서 어텐션을 수행하지 않아 속도 향상
    for seq in input_ids:
        seq_mask = [float(i > 0) for i in seq]
        attention_masks.append(seq_mask)

    # 데이터를 파이토치의 텐서로 변환
    inputs = torch.tensor(input_ids)
    masks = torch.tensor(attention_masks)

    return inputs, masks


def test_sentences(sentences):
    # 평가모드로 변경
    model.eval()

    # 문장을 입력 데이터로 변환
    inputs, masks = convert_input_data(sentences)

    # 데이터를 GPU에 넣음
    b_input_ids = inputs.to(device)
    b_input_mask = masks.to(device)

    # 그래디언트 계산 안함
    with torch.no_grad():
        # Forward 수행
        outputs = model(b_input_ids,
                        token_type_ids=None,
                        attention_mask=b_input_mask)

    # 로스 구함
    logits = outputs[0]

    # CPU로 데이터 이동
    logits = logits.detach().cpu().numpy()

    return int(np.argmax(logits))

test_sentences(['연기는 별로지만 재미 하나는 끝내줌!'])

######
def read_pickle(file_name: str) -> pd.DataFrame:
    return pd.read_pickle('Pickle/' + file_name)

video_comment = read_pickle('video_comment.pkl')

df_comment = video_comment[['Comment']]
df_comment['Comment'] = [re.sub('[^가-힣 ]', '', s) for s in df_comment['Comment']]
df_comment['Comment'] = df_comment.Comment.str.strip()
idx = df_comment[df_comment['Comment'] == ''].index
df_comment = df_comment.drop(idx).reset_index(drop=True)

# Predict
predict = []
for i in range(len(df_comment)):
    score = test_sentences([df_comment['Comment'][i]])
    predict.append(score)

def _replaceitem(x):
    if x == 0:
        return("공포")
    elif x == 1:
        return("놀람")
    elif x == 2:
        return("분노")
    elif x == 3:
        return("슬픔")
    elif x == 4:
        return("중립")
    elif x == 5:
        return("행복")
    else:
        return("혐오")

result = pd.DataFrame(predict)
result = result[0].apply(pd.Series)
result = result.merge(df_comment, left_index=True, right_index=True)
result = result.rename(columns = {0: 'emotion', 'Comment': 'comment'}, inplace = False)
print("hi")

result['emotion'] = result['emotion'].apply(_replaceitem)
print("hi")

test = result[result['emotion']=='공포']