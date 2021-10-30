# ====================== LIBRARY SETUP ====================== #
import pandas as pd
import re
from sklearn.model_selection import train_test_split
import numpy as np
import random
import time
import datetime

# Torch
import torch, gc
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences

# ====================== LOADING FINE-TUNING DATA ====================== #
# Fine-tuning-1
df_a = pd.read_csv('data/한국어_단발성_대화_데이터셋.csv')
# Fine-tuning-2
df_b = pd.read_csv('data/한국어_연속적_대화_데이터셋.csv')
df_b = df_b.iloc[:,1:3]
df_b = df_b.rename(columns={'Unnamed: 1': 'Sentence', 'Unnamed: 2': 'Emotion'}, inplace=False)
# Merge
tuning_data = df_a.append(df_b, ignore_index=True)

# Change Column Name
tuning_data = tuning_data.rename(columns={'Emotion': 'emotion', 'Sentence': 'sentence'}, inplace=False)

# Converting emotion to numeric figure
tuning_data.loc[(tuning_data['emotion'] == "공포"), 'emotion'] = 0  #공포 => 0
tuning_data.loc[(tuning_data['emotion'] == "놀람"), 'emotion'] = 1  #놀람 => 1
tuning_data.loc[(tuning_data['emotion'] == "분노"), 'emotion'] = 2  #분노 => 2
tuning_data.loc[(tuning_data['emotion'] == "슬픔"), 'emotion'] = 3  #슬픔 => 3
tuning_data.loc[(tuning_data['emotion'] == "중립"), 'emotion'] = 4  #중립 => 4
tuning_data.loc[(tuning_data['emotion'] == "행복"), 'emotion'] = 5  #행복 => 5
tuning_data.loc[(tuning_data['emotion'] == "혐오"), 'emotion'] = 6  #혐오 => 6

# Filtering unlabelled emotions in dataframe
filter = tuning_data['emotion'].isin([0, 1, 2, 3, 4, 5, 6])
tuning_data = tuning_data[filter]

# Train & Test
tuning_data['emotion'] = pd.to_numeric(tuning_data['emotion'])
dataset_train, dataset_test = train_test_split(tuning_data, test_size=0.2, random_state=0)

# ====================== BERT SETUP ====================== #
# Sentence & Label
sentences = dataset_train['sentence']
sentences = ["[CLS] " + str(sentence) + " [SEP]" for sentence in sentences]

labels = dataset_train['emotion'].values

# Tokenization
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]

# Set the max length of sentence
max_len = 128
# Convert Tokens into index(array)
input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
# 문장을 MAX_LEN 길이에 맞게 자르고, 모자란 부분을 패딩 0으로 채움
input_ids = pad_sequences(input_ids, maxlen=max_len, dtype="long", truncating="post", padding="post")

# Set attention mask
attention_masks = []
# 어텐션 마스크를 패딩이 아니면 1, 패딩이면 0으로 설정
# 패딩 부분은 BERT 모델에서 어텐션을 수행하지 않아 속도 향상
for seq in input_ids:
    seq_mask = [float(i>0) for i in seq]
    attention_masks.append(seq_mask)

# Trainin Validation Set
train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids,
                                                                                    labels,
                                                                                    random_state=2018,
                                                                                    test_size=0.1)
# Training validation set in attention masks
train_masks, validation_masks, _, _ = train_test_split(attention_masks,
                                                       input_ids,
                                                       random_state=2018,
                                                       test_size=0.1)
# Converting data into pytorch tensor
train_inputs = torch.tensor(train_inputs)
train_labels = torch.tensor(train_labels)
train_masks = torch.tensor(train_masks)
validation_inputs = torch.tensor(validation_inputs)
validation_labels = torch.tensor(validation_labels)
validation_masks = torch.tensor(validation_masks)

# Batch size
batch_size = 8

# Pytorch dataloader with data, masks and labels
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

# Device (GPU or CPU)
if torch.cuda.is_available():
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print('No GPU available, using the CPU instead.')

# Model Setup
model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=7)
model.to(device)

# Optimizer
optimizer = AdamW(model.parameters(),
                  lr=2e-5, # Learning rate
                  eps=1e-8 # 0으로 나누는 것을 방지하기 위한 epsilon 값
                  )
# Number of epoch
epochs = 3
# Total Steps : number of batch in dataset * epoch
total_steps = len(train_dataloader) * epochs
# 처음에 학습률을 조금씩 변화시키는 스케줄러 생성
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0,
                                            num_training_steps=total_steps)

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()

    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def format_time(elapsed):
    # Round Up
    elapsed_rounded = int(round((elapsed)))
    # Time format hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

# 재현을 위해 랜덤시드 고정
seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
if device == 'gpu':
    torch.cuda.manual_seed_all(seed_val)
else:
    print("Your device is running CPU, manual seed will be used instead")

# Reset gradient
model.zero_grad()

# ====================== RUNNING MODEL ====================== #
# ====================== MODEL TRAINING ====================== #
# Run model in epochs
for epoch_i in range(0, epochs):
    # ======================================== #
    #               Training                   #
    # ======================================== #
    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')
    # Start time
    t0 = time.time()
    # Total loss reset
    total_loss = 0
    # Pytorch training model
    model.train()
    # Loading dataloader in batch size repeatedly
    for step, batch in enumerate(train_dataloader):
        # Set result display
        if step % 500 == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
        # Inserting batch into GPU
        batch = tuple(t.to(device) for t in batch)
        # Extracting data from batch
        b_input_ids, b_input_mask, b_labels = batch
        # Running Forward
        outputs = model(b_input_ids,
                        token_type_ids=None,
                        attention_mask=b_input_mask,
                        labels=b_labels)
        # Calculating loss
        loss = outputs[0]
        # Calculating total loss
        total_loss += loss.item()
        # Running Backward to calculate gradient
        loss.backward()
        # Clipping gradient
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # 그래디언트를 통해 가중치 파라미터 업데이트
        optimizer.step()
        # 스케줄러로 학습률 감소
        scheduler.step()
        # Reset gradient
        model.zero_grad()
    # Calculating average loss
    avg_train_loss = total_loss / len(train_dataloader)
    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epcoh took: {:}".format(format_time(time.time() - t0)))
    # ======================================== #
    #               Validation                 #
    # ======================================== #
    print("")
    print("Running Validation...")
    # Start time
    t0 = time.time()
    # Pytorch evaluating model
    model.eval()
    # Reset inputs
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    # Loading dataloader in batch size repeatedly
    for batch in validation_dataloader:
        # Inserting batch into GPU
        batch = tuple(t.to(device) for t in batch)
        # Extracting data from batch
        b_input_ids, b_input_mask, b_labels = batch
        # Excluding gradient calculation
        with torch.no_grad():
            # Running Forward
            outputs = model(b_input_ids,
                            token_type_ids=None,
                            attention_mask=b_input_mask)
        # Calculating loss
        logits = outputs[0]
        # Transferring data into CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        # Calculating accuracy using output logits and labels
        tmp_eval_accuracy = flat_accuracy(logits, label_ids)
        eval_accuracy += tmp_eval_accuracy
        nb_eval_steps += 1
    print("  Accuracy: {0:.2f}".format(eval_accuracy / nb_eval_steps))
    print("  Validation took: {:}".format(format_time(time.time() - t0)))
print("")
print("Training complete!")

# ====================== MODEL TESTING ====================== #

# Sentence & Label
sentences = dataset_test['sentence']
sentences = ["[CLS] " + str(sentence) + " [SEP]" for sentence in sentences]

labels = dataset_test['emotion'].values

# Tokenization
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]

# Set the max length of sentence
max_len = 128
# Convert Tokens into index(array)
input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
# 문장을 MAX_LEN 길이에 맞게 자르고, 모자란 부분을 패딩 0으로 채움
input_ids = pad_sequences(input_ids, maxlen=max_len, dtype="long", truncating="post", padding="post")

attention_masks = []
# 어텐션 마스크를 패딩이 아니면 1, 패딩이면 0으로 설정
# 패딩 부분은 BERT 모델에서 어텐션을 수행하지 않아 속도 향상
for seq in input_ids:
    seq_mask = [float(i > 0) for i in seq]
    attention_masks.append(seq_mask)

# Converting data into pytorch tensor
test_inputs = torch.tensor(input_ids)
test_labels = torch.tensor(labels)
test_masks = torch.tensor(attention_masks)

# Pytorch dataloader with data, masks and labels
test_data = TensorDataset(test_inputs, test_masks, test_labels)
test_sampler = RandomSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

# Start time
t0 = time.time()
# Pytorch evaluating model
model.eval()
# Reset inputs
eval_loss, eval_accuracy = 0, 0
nb_eval_steps, nb_eval_examples = 0, 0
# Loading dataloader in batch size repeatedly
for step, batch in enumerate(test_dataloader):
    # Display results
    if step % 100 == 0 and not step == 0:
        elapsed = format_time(time.time() - t0)
        print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(test_dataloader), elapsed))

    # Inserting batch into GPU
    batch = tuple(t.to(device) for t in batch)
    # Extracting data from batch
    b_input_ids, b_input_mask, b_labels = batch
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
    label_ids = b_labels.to('cpu').numpy()
    # Calculating accuracy using output logits and labels
    tmp_eval_accuracy = flat_accuracy(logits, label_ids)
    eval_accuracy += tmp_eval_accuracy
    nb_eval_steps += 1

print("")
print("Accuracy: {0:.2f}".format(eval_accuracy / nb_eval_steps))
print("Test took: {:}".format(format_time(time.time() - t0)))

print("Completed Running Test if you see this message")

# ====================== MODEL EXPORTING ====================== #
# ======================================================== #
torch.save(model.state_dict(), 'bert_model_gpu.pth')
# ======================================================== #