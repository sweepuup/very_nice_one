import torch
import math
from torch import nn
import torch.nn.functional as F
import Tokenizer
from datasets import load_dataset
import time
import json
from transformers import AdamW, get_scheduler
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence


### TOKENIZER ##########################################################################################################
vocabulary = Tokenizer.get_vocabulary()
token_vocabulary = Tokenizer.get_token_vocabulary()


### TRANSFORMER ########################################################################################################
d_model = 384
num_heads = 6
drop_prob = 0.1
batch_size = 38  # batch_size must be divisible by num_heads / len(train_input) must be divisible by batch_size
max_sequence_length = 256
ffn_hidden = d_model * 4
num_layers = 6
save_path = 'models/my_model.pt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    scaled = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_k)
    if mask is not None:
        scaled += mask.to(device)
    attention = F.softmax(scaled, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv_layer = nn.Linear(d_model, 3 * d_model)
        self.linear_layer = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        batch_size, max_sequence_length, d_model = x.size()
        qkv = self.qkv_layer(x)
        qkv = qkv.reshape(batch_size, max_sequence_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=-1)
        values, attention = scaled_dot_product(q, k, v, mask)
        values = values.reshape(batch_size, max_sequence_length, self.num_heads * self.head_dim)
        out = self.linear_layer(values)
        return out


class LayerNormalization(nn.Module):
    def __init__(self, parameters_shape, eps=1e-5):
        super().__init__()
        self.parameters_shape = parameters_shape
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(parameters_shape))
        self.beta = nn.Parameter(torch.zeros(parameters_shape))

    def forward(self, inputs):
        dims = [-(i + 1) for i in range(len(self.parameters_shape))]
        mean = inputs.mean(dim=dims, keepdim=True)
        var = ((inputs - mean) ** 2).mean(dim=dims, keepdim=True)
        std = (var + self.eps).sqrt()
        y = (inputs - mean) / std
        out = self.gamma * y + self.beta
        return out


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

    def forward(self, sequence_length):
        even_i = torch.arange(0, self.d_model, 2).float()
        denominator = torch.pow(10000, even_i / self.d_model)
        position = torch.arange(sequence_length).reshape(sequence_length, 1)
        even_PE = torch.sin(position / denominator)
        odd_PE = torch.cos(position / denominator)
        stacked = torch.stack([even_PE, odd_PE], dim=2)
        PE = torch.flatten(stacked, start_dim=1, end_dim=2)
        return PE


class TransformerLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob):
        super(TransformerLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.norm1 = LayerNormalization(parameters_shape=[d_model])
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = LayerNormalization(parameters_shape=[d_model])
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x, original_inputs):
        input_pad_mask = (original_inputs != 0)
        index = torch.argmax(input_pad_mask.sum(dim=1))
        max_length = 0
        for element in original_inputs[index]:
            if element != 0:
                max_length += 1
            else:
                break
        seq_len = x.size()[1]
        causal_mask = torch.tril(torch.ones(seq_len, seq_len))
        mask = torch.where(causal_mask == 0, torch.tensor(float('-inf')), causal_mask)
        mask[mask == 1] = 0
        mask[max_length:, max_length:] = float('-inf')

        residual_x = x
        x = self.attention(x, mask=mask)
        # x = self.dropout1(x)
        x = self.norm1(x + residual_x)
        residual_x = x
        x = self.ffn(x)
        # x = self.dropout2(x)
        x = self.norm2(x + residual_x)
        return x


class SequentialTransformer(nn.Sequential):
    def forward(self, *inputs):
        x, original_inputs = inputs
        for module in self._modules.values():
            new_x = module(x, original_inputs)
        return new_x


class Transformer(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob, num_layers):
        super().__init__()
        self.d_model = d_model
        self.token_embedding = nn.Embedding(len(vocabulary), d_model)
        # self.token_embedding = nn.Embedding(len(true_vocabulary), d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        self.layers = SequentialTransformer(*[TransformerLayer(d_model, ffn_hidden, num_heads, drop_prob)
                                              for _ in range(num_layers)])
        self.output_layers = nn.Linear(d_model, len(vocabulary))
        # self.output_layers = nn.Linear(d_model, len(true_vocabulary))

    def forward(self, x, targets):
        original_inputs = x
        token_embeddings = self.token_embedding(x) * math.sqrt(self.d_model)
        pos_encoding = self.positional_encoding(x.size()[1]).to(device).unsqueeze(0).repeat(x.size(0), 1, 1)
        x = token_embeddings + pos_encoding
        x = self.layers(x, original_inputs)
        logits = self.output_layers(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def generate(self, x):
        original_inputs = x
        token_embeddings = self.token_embedding(x) * math.sqrt(self.d_model)
        pos_encoding = self.positional_encoding(x.size()[1]).to(device).unsqueeze(0).repeat(x.size(0), 1, 1)
        x = token_embeddings + pos_encoding
        x = self.layers(x, original_inputs)
        x = self.output_layers(x)
        return F.softmax(x, dim=-1)


### DATA PREPROCESSING #################################################################################################
print('Data Preprocessing...')
start_time = time.time()

def save_tokenized_data(name, tokenized_dataset):
    with open(name, 'w') as file:
        json.dump(tokenized_dataset, file)

def load_tokenized_data(name):
    with open(f'tokenized_datasets/{name}', 'r') as file:
        loaded_tokenized_data = json.load(file)
    return loaded_tokenized_data

# raw_dataset = load_dataset('c4', 'realnewslike')  #***********************************#
# raw_dataset = raw_dataset['train'].select(range(round(len(raw_dataset['train']) / 1000)))
# raw_dataset = [Tokenizer.tokenize_sequence(raw_dataset['text'][i]) for i in range(len(raw_dataset['text']))]
# save_tokenized_data('tokenized_datasets/c4_realnewslike.json', raw_dataset)

raw_dataset = load_tokenized_data('c4_realnewslike.json')  #***********************************#
token_dataset = []
for i in range(len(raw_dataset)):
    for j in range(len(raw_dataset[i])):
        token_dataset.append(raw_dataset[i][j])
token_dataset = token_dataset[:round(max_sequence_length * math.floor(len(token_dataset) / max_sequence_length))]
train_input = [[] for i in range(math.floor(len(token_dataset) / (max_sequence_length * 2)))]
train_output = [[] for i in range(math.floor(len(token_dataset) / (max_sequence_length * 2)))]
for i in range(0, len(token_dataset) - max_sequence_length, max_sequence_length * 2):
    for j in range(max_sequence_length):
        train_input[round(i / (max_sequence_length * 2))].append(token_dataset[i + j])
        train_output[round(i / (max_sequence_length * 2))].append(token_dataset[i + j + max_sequence_length])
print(f'len(train_input) = {len(train_input)}')

# # raw_train_dataset, raw_eval_dataset = train_test_split(raw_dataset['train'].select(range(round(len(raw_dataset['train']) / 25))), test_size=0.2)
train_input = [seq[:max_sequence_length] if len(seq) > max_sequence_length else seq for seq in train_input]
train_output = [seq[:max_sequence_length] if len(seq) > max_sequence_length else seq for seq in train_output]
train_input = [torch.tensor(seq, dtype=torch.long) for seq in train_input]
train_output = [torch.tensor(seq, dtype=torch.long) for seq in train_output]
# train_input = [Tokenizer.pad_to_length(seq, max_sequence_length) for seq in train_input]
# train_output = [Tokenizer.pad_to_length(seq, max_sequence_length) for seq in train_output]
train_dataset = [(train_input[i], train_output[i]) for i in range(len(train_input))]
# train_dataset = [pad_sequence(train_dataset[i], batch_first=True, padding_value=0) for i in range(len(train_dataset))]
train_batch = [[[] for i in range(round(len(train_dataset) / batch_size))] for j in range(2)]
train_batch_count = 0
for i in range(0, len(train_dataset), batch_size):
    for j in range(batch_size):
        train_batch[0][train_batch_count].append(train_dataset[i + j][0])
        train_batch[1][train_batch_count].append(train_dataset[i + j][1])
    train_batch_count += 1


### TRAINING ###########################################################################################################
print('Training...')
model = Transformer(d_model, ffn_hidden, num_heads, drop_prob, num_layers)
print(f'model parameters: {sum(p.numel() for p in model.parameters())}')
model.to(device)
epochs = 5
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

num_training_steps = epochs * len(train_dataset)
lr_scheduler = get_scheduler(
    name='linear',
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)
train_epoch_average_loss = []
train_loss_total = 0
for epoch in range(epochs):
    model.train()
    train_loss = 0
    for i in range(len(train_batch[0])):
        inputs = torch.stack(train_batch[0][i]).to(device)
        labels = torch.stack(train_batch[1][i]).to(device)
        logits, loss = model.forward(inputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        train_loss_total += loss
        if i % 10 == 0:
            print('TRAINING...')
            print(f'EPOCH {epoch}, batch {i}/{len(train_batch[0])}')
            print(f'loss: {loss}')
    train_epoch_average_loss.append((train_loss_total / len(train_batch[0])))
    train_loss_total = 0
    # model.eval()
    # eval_loss = 0
    # with torch.no_grad():
    #     for i, batch in enumerate(eval_dataset):
    #         inputs = batch[0].unsqueeze(0).to(device)
    #         labels = batch[1].unsqueeze(0).to(device)
    #         logits, loss = model(inputs, labels)
    #         if i % 10 == 0:
    #             print('EVALUATING...')
    #             print(f'EPOCH {epoch}, batch {i}/{len(eval_dataset)}')
    #             print(f'loss: {loss}')
for i in range(len(train_epoch_average_loss)):
    print(f'EPOCH {i} AVERAGE LOSS: {train_epoch_average_loss[i]}')
torch.save(model.state_dict(), save_path)


end_time = time.time()
total_time = end_time - start_time
print(f'{total_time} seconds')