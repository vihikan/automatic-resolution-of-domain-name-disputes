import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim


sys.path.append('/home/nlp/wipo_common/data_loader')
from data_loader import get_text_and_label, get_data_from_json
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel


## ===============================================================================
notebook_dir = '/home/nlp/'
project_files_dir = '/home/nlp/json_files'
wipo_text_path = os.path.join(project_files_dir, "wipo_cases_preprocessed_bert.json")

wipo_dict = get_data_from_json(wipo_text_path)
cases_sentences, labels, label_transfer, label_denied, label_cancel = get_text_and_label(wipo_dict)

print('Total data: ' + str(len(cases_sentences)))
print('Label transfer: ' + str(len(label_transfer)))
print('Label denied: ' + str(len(label_denied)))
print('Label cancellation: ' + str(len(label_cancel)))
print()

## ===============================================================================

cases = dict()
for key, val in cases_sentences.items():
    cases[key] = ' '.join(val)

print(cases['case-0'])

## ===============================================================================

temp_labels = labels
for key, val in temp_labels.items():
    if val == "transfer" or val == "cancellation":
        labels[key] = "success"
    elif val == "complaint denied":
        labels[key] = "failure"

## ======================================================================================================

# Making the training, dev, test set
label_list = [*labels]

# Train cases are first 80% of the all cases
train_cases = label_list[-24249:]
dev_test_cases = label_list[:6062]

labels_dev_test = dict()

dev_test_success = []
dev_test_failure = []

for case in dev_test_cases:
    if wipo_dict[case]['status']  == 'transfer':
        dev_test_success.append(case)
    elif wipo_dict[case]['status'] == 'complaint denied':
        dev_test_failure.append(case)
    elif wipo_dict[case]['status'] == 'cancellation':
        dev_test_success.append(case)

print('Dev Test Success : ' + str(len(dev_test_success)))
print('Dev Test Failure : ' + str(len(dev_test_failure)))

## ===============================================================================

dev_cases = dev_test_success[-2848:] + dev_test_failure[-183:]
test_cases = dev_test_success[:2848] + dev_test_failure[:183]
## ===============================================================================

train_text = []
train_label = []
dev_text = []
dev_label = []
test_text = []
test_label = []

for case in train_cases:
    train_text.append(cases[case].lower())
    train_label.append(labels[case])
    
for case in dev_cases:
    dev_text.append(cases[case].lower())
    dev_label.append(labels[case])
    
for case in test_cases:
    test_text.append(cases[case].lower())
    test_label.append(labels[case])

le = LabelEncoder()
le.fit(train_label)

y_train = le.fit_transform(train_label).astype('float')
y_dev = le.fit_transform(dev_label).astype('float')
y_test = le.fit_transform(test_label).astype('float')

print(list(le.classes_))
print(str(y_train[0]) + ' = ' + train_text[0])
print()
print(str(y_dev[0]) + ' = ' + dev_text[0])
print()
print(str(y_test[0]) + ' = ' + test_text[0])

## ===============================================================================

train_data = train_text
dev_data = dev_text
test_data = test_text

## ===============================================================================

class WIPODataset(Dataset):

    def __init__(self, texts, labels, maxlen):

        self.texts = texts
        self.labels = labels

        #Initialize the BERT tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")

        self.maxlen = maxlen

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):

        #Selecting the sentence and label at the specified index in the data frame
        sentence = self.texts[index]
        label = self.labels[index]

        #Preprocessing the text to be suitable for BERT
        tokens = self.tokenizer.tokenize(sentence) #Tokenize the sentence
        tokens = ['[CLS]'] + tokens + ['[SEP]'] #Insering the CLS and SEP token in the beginning and end of the sentence
        if len(tokens) < self.maxlen:
            tokens = tokens + ['[PAD]' for _ in range(self.maxlen - len(tokens))] #Padding sentences
        elif len(tokens) > self.maxlen:
            tokens = ['[CLS]'] + tokens[-(self.maxlen-1):] #Prunning the list to be of specified max length

        tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens) #Obtaining the indices of the tokens in the BERT Vocabulary
        tokens_ids_tensor = torch.tensor(tokens_ids) #Converting the list to a pytorch tensor

        #Obtaining the attention mask i.e a tensor containing 1s for no padded tokens and 0s for padded ones
        attn_mask = (tokens_ids_tensor != 0).long()

        return tokens_ids_tensor, attn_mask, label


## ===============================================================================

maxlen = 512

train_set = WIPODataset(train_data, y_train, maxlen = maxlen)
dev_set = WIPODataset(dev_data, y_dev, maxlen = maxlen)
test_set = WIPODataset(test_data, y_test, maxlen = maxlen)

#Creating intsances of training and development dataloaders
batch_size = 8
num_workers = 5
train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=num_workers)
dev_loader = DataLoader(dev_set, batch_size=batch_size, num_workers=num_workers)
test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=num_workers)

print("Done preprocessing training and development data.")

## ===============================================================================

class WIPOClassifier(nn.Module):

    def __init__(self):
        super(WIPOClassifier, self).__init__()
        #Instantiating BERT model object 
        self.bert_layer = AutoModel.from_pretrained('nlpaueb/legal-bert-base-uncased')

        # freezing the embedding
        modules = [self.bert_layer.embeddings, *self.bert_layer.encoder.layer[:8]]
        for module in modules:
            for param in module.parameters():
                param.requires_grad = False
        
        #Classification layer
        #input dimension is 768 because [CLS] embedding has a dimension of 768
        #output dimension is 2 because we're working with 2 labels
        self.cls_layer1 = nn.Linear(768, 64)
        self.cls_dropout = nn.Dropout(0.2)
        self.cls_layer2 = nn.Linear(64, 2)

    def forward(self, seq, attn_masks):
        '''
        Inputs:
            -seq : Tensor of shape [B, T] containing token ids of sequences
            -attn_masks : Tensor of shape [B, T] containing attention masks to be used to avoid contibution of PAD tokens
        '''

        #Feeding the input to BERT model to obtain contextualized representations
        cont_reps, _ = self.bert_layer(seq, attention_mask = attn_masks)

        #Obtaining the representation of [CLS] head (the first token)
        cls_rep = cont_reps[:, 0]

        # cls_rep = torch.reshape(cls_rep,(8,1,768))
        #Feeding cls_rep to the classifier layer
        logits1 = self.cls_layer1(cls_rep)
        # logits1 = self.lstm_layer1(cls_rep)

        logits_d1 = self.cls_dropout(logits1)

        logits2 = self.cls_layer2(logits_d1)

        return logits2

## ===============================================================================


## ===============================================================================

def train(net, criterion, opti, train_loader, dev_loader, max_eps, gpu):

    best_acc = 0
    best_prec = 0
    best_rec = 0
    best_f1 = 0
    best_loss = 999

    # Early stopping params
    patience = 3
    cur_patience = patience
    min_delta = 0.0001

    st = time.time()
    for ep in range(1, max_eps+1):
        net.train()
        
        for it, (seq, attn_masks, labels) in enumerate(train_loader):
            #Clear gradients
            opti.zero_grad()  
            #Converting these to cuda tensors
            seq, attn_masks, labels = seq.cuda(gpu), attn_masks.cuda(gpu), labels.cuda(gpu)

            #Obtaining the logits from the model
            logits = net(seq, attn_masks)

            #Computing loss
            loss = criterion(logits.squeeze(-1), labels.long())

            #Backpropagating the gradients
            loss.backward()

            #Optimization step
            opti.step()
              
            if it > 1 and it % 100 == 0:
                
                acc = get_accuracy_from_logits(logits, labels)
                print("Iteration {} of epoch {} complete. Loss: {:.3f}; Accuracy: {:.3f}; Time taken (s): {}".format(it, ep, loss.item(), acc, (time.time()-st)))
    
                st = time.time()

        
        dev_acc, dev_f1_ma, dev_f1_mi, dev_prec_ma, dev_prec_mi, dev_recall_ma, dev_recall_mi, dev_loss, _ = evaluate(net, criterion, dev_loader, gpu)
        print()
        print("Development score: Macro / Micro")
        print()
        print("Loss: {:12.3f}".format(dev_loss))
        print("Accuracy: {:8.3f}".format(dev_acc))
        print("F1: {:14.3f} / {:2.3f}".format(dev_f1_ma, dev_f1_mi))
        print("Precision: {:7.3f} / {:2.3f}".format(dev_prec_ma, dev_prec_mi))
        print("Recall: {:10.3f} / {:2.3f}".format(dev_recall_ma, dev_recall_mi))

        
        if dev_loss < best_loss:
            print("Best development loss improved from {} to {}, saving model...".format(best_loss, dev_loss))
            best_loss = dev_loss
            torch.save(net.state_dict(), checkpoint_loss_name)

        if dev_f1_ma > best_f1:
            # Early stopping
            if (dev_f1_ma - best_f1) < min_delta:
                cur_patience -= 1
            else:
                cur_patience = patience

            print("Best development f1 improved from {} to {}, saving model...".format(best_f1, dev_f1_ma))
            best_f1 = dev_f1_ma
            torch.save(net.state_dict(), checkpoint_f1_name)
        else:
            cur_patience -= 1

        if cur_patience <= 0:
            break
        
        print()

## ===============================================================================

def get_accuracy_from_logits(logits, labels):
    probs = torch.softmax(logits.unsqueeze(-1), dim=1)
    soft_probs = probs.argmax(dim=1)
    acc = (soft_probs.squeeze() == labels).float().mean()
    return acc

def get_precision_from_logits(logits, labels):
    probs = torch.softmax(logits.unsqueeze(-1), dim=1)
    soft_probs = probs.argmax(dim=1)

    soft_probs = soft_probs.detach().cpu().numpy()
    labels = labels.to('cpu').numpy()

    pred_flat = soft_probs.flatten()
    labels_flat = labels.flatten()
    return precision_score(labels_flat, pred_flat,  average='macro'), precision_score(labels_flat, pred_flat,  average='micro')

def get_recall_from_logits(logits, labels):
    probs = torch.softmax(logits.unsqueeze(-1), dim=1)
    soft_probs = probs.argmax(dim=1)

    soft_probs = soft_probs.detach().cpu().numpy()
    labels = labels.to('cpu').numpy()

    pred_flat = soft_probs.flatten()
    labels_flat = labels.flatten()
    return recall_score(labels_flat, pred_flat,  average='macro'), recall_score(labels_flat, pred_flat,  average='micro')

def get_f1_from_logits(logits, labels):
  # Move logits and labels to CPU
    probs = torch.softmax(logits.unsqueeze(-1), dim=1)
    soft_probs = probs.argmax(dim=1)

    soft_probs = soft_probs.detach().cpu().numpy()
    labels = labels.to('cpu').numpy()

    pred_flat = soft_probs.flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, pred_flat, average='macro'), f1_score(labels_flat, pred_flat, average='micro')

def evaluate(net, criterion, dataloader, gpu):
    net.eval()

    mean_acc, mean_loss, mean_f1_ma, mean_f1_mi, mean_precision_ma, mean_precision_mi, mean_recall_ma, mean_recall_mi = 0, 0, 0, 0, 0, 0, 0, 0
    count = 0
    predicted_labels = []
    logits_all = None
    labels_all = None

    with torch.no_grad():
        for seq, attn_masks, labels in dataloader:
            seq, attn_masks, labels = seq.cuda(gpu), attn_masks.cuda(gpu), labels.cuda(gpu)
            logits = net(seq, attn_masks)
            
            if logits_all == None:
                logits_all = logits
                labels_all = labels
            else:
                logits_all = torch.cat((logits_all, logits), 0)
                labels_all = torch.cat((labels_all, labels), 0)

            count += 1

        mean_loss = criterion(logits_all.squeeze(-1), labels_all.long()).item()

        mean_acc = get_accuracy_from_logits(logits_all, labels_all)

        f1_ma, f1_mi = get_f1_from_logits(logits_all, labels_all)
        mean_f1_ma = f1_ma 
        mean_f1_mi = f1_mi

        prec_ma, prec_mi = get_precision_from_logits(logits_all, labels_all)
        mean_precision_ma = prec_ma
        mean_precision_mi = prec_mi 
            
        recall_ma, recall_mi = get_recall_from_logits(logits_all, labels_all)
        mean_recall_ma = recall_ma
        mean_recall_mi = recall_mi
        
        probs = torch.softmax(logits_all.unsqueeze(-1), dim=1)
        soft_probs = probs.argmax(dim=1)
        soft_probs = soft_probs.detach().cpu().numpy()
        pred = soft_probs.flatten()
        predicted_labels += pred.tolist()

    return mean_acc, mean_f1_ma, mean_f1_mi, mean_precision_ma, mean_precision_mi, mean_recall_ma, mean_recall_mi, mean_loss, predicted_labels


## ===============================================================================


## ===============================================================================

# Use tmux to run on server with multiple GPUs
# Then set this env var: export CUDA_VISIBLE_DEVICES=5
# PyTorch will read all GPU as device 0.
gpu = 0 #gpu ID
trying_number = 5
num_epoch = 20
models_save_name = "paper/models/model_legalbert_freeze_emb_8enc_earlystop_"

for t in range(trying_number):
    print()
    print("Training model part " + str(t))
    print()

    model_save_dir = models_save_name + str(t)
    if not os.path.exists(os.path.join(notebook_dir, model_save_dir)):
        os.makedirs(os.path.join(notebook_dir, model_save_dir))

    checkpoint_loss_name = os.path.join(notebook_dir, model_save_dir+"/bert_context_v2_1_best_loss.pt")
    checkpoint_f1_name = os.path.join(notebook_dir, model_save_dir+"/bert_context_v2_1_best_f1.pt")

    print("Creating the classifier, initialised with pretrained BERT-BASE parameters...")
    net = WIPOClassifier()
    net.cuda(gpu) #Enable gpu support for the model
    print("Done creating the classifier.")

    ## ===============================================================================

    criterion = nn.CrossEntropyLoss()
    opti = optim.Adam(net.parameters(), lr = 1e-5)

    ## ===============================================================================

    train(net, criterion, opti, train_loader, dev_loader, num_epoch, gpu)

    print()
    print("Training model part " + str(t) + " is done.")
    print()
    print("Resetting PyTorch...")

    del net
    del criterion
    del opti
    
    try:
        torch.cuda.empty_cache()
    except RuntimeError as e:
        sys.exit(str(e))