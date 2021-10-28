# ====================== LIBRARY SETUP ====================== #
import pandas as pd
import re
import torch
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW, BertConfig
from keras.preprocessing.sequence import pad_sequences
import numpy as np

# ====================== FUNCTION SETUP ====================== #
#print(torch.cuda.memory_allocated())
#print(torch.cuda.memory_reserved())
def read_pickle(file_name: str) -> pd.DataFrame:
    return pd.read_pickle('Pickle/' + file_name)

def sentiment_score(x):
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

def setup_device():
    # Setup device (CPU or GPU)
    if torch.cuda.is_available():
        # Remove memory
        torch.cuda.empty_cache()
        # Set GPU
        device = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))
    else:
        # Set CPU
        device = torch.device("cpu")
        print('No GPU available, using the CPU instead.')
    return device

def load_model(path, device):
    """
    path : model file name
    """
    # Model setup
    model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=7)
    model.load_state_dict(torch.load(path, map_location='cpu'))
    model.to(device)
    model.eval()

    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)

    return model, tokenizer

def run_model(tokenizer, sentences, device):
    """
    sentences : put sentences extracted from youtube. Please note that the sentence should be in the nested list format
    device : run setup_device() function
    """
    # Pytorch evaluating model
    model.eval()

    # Tokenization
    tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
    # Set the max length of sentence
    MAX_LEN = 128
    # Convert Tokens into index(array)
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
    # 문장을 MAX_LEN 길이에 맞게 자르고, 모자란 부분을 패딩 0으로 채움
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

    # Set attention mask
    attention_masks = []
    # 어텐션 마스크를 패딩이 아니면 1, 패딩이면 0으로 설정
    # 패딩 부분은 BERT 모델에서 어텐션을 수행하지 않아 속도 향상
    for seq in input_ids:
        seq_mask = [float(i > 0) for i in seq]
        attention_masks.append(seq_mask)

    # Converting data into pytorch tensor
    inputs = torch.tensor(input_ids)
    masks = torch.tensor(attention_masks)
    # Inserting batch into GPU
    b_input_ids = inputs.to(device)
    b_input_mask = masks.to(device)
    # No gradient calculation
    with torch.no_grad():
        # Running Forward
        outputs = model(b_input_ids,
                        token_type_ids=None,
                        attention_mask=b_input_mask)

    # Calculating loss
    logits = outputs[0]
    # Transferring data into CPU
    logits = logits.detach().cpu().numpy()

    return int(np.argmax(logits))

def run_sentiment():

    # Read comment pickle after running youtube comment api
    df = read_pickle('video_comment.pkl')
    print("Successfully loaded video comments")

    # Cleaning comment after the extraction
    df_comment = df[['comment']]
    df_comment['comment'] = [re.sub('[^가-힣 ]', '', s) for s in df_comment['comment']]
    df_comment['comment'] = df_comment.comment.str.strip()
    idx = df_comment[df_comment['comment'] == ''].index
    df_comment = df_comment.drop(idx).reset_index(drop=True)

    # Perform sentiment analysis using model
    print("Started running sentiment analysis")
    predict = []
    for i in range(len(df_comment)):
        score = run_model(sentences=[df_comment['comment'][i]], tokenizer=tokenizer, device=device)
        predict.append(score)

    # Converting sentiment scores to language
    # result[0] has to be adjusted in the future in API
    print("Started putting results into dataframe")
    result = pd.DataFrame(predict)
    result = result[0].apply(pd.Series)
    result = result.merge(df_comment, left_index=True, right_index=True)
    result = result.rename(columns={0: 'emotion'}, inplace=False)
    result['emotion'] = result['emotion'].apply(sentiment_score)

    return result

# ====================== RUNNING MODEL ====================== #
device = setup_device()
model, tokenizer = load_model(path='bert_model_gpu.pth', device=device)
result = run_sentiment()

# ====================== ANALYSIS ====================== #
공포 = result[result['emotion']=='공포']
혐오 = result[result['emotion']=='혐오']