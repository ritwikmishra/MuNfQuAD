# model libs
import os, torch, numpy as np, random, time
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers.modeling_outputs import SequenceClassifierOutput

# data libs
import pickle, glob, json, re
from tqdm import tqdm

hyper_params = {
    "run_ID": "17c3", # "17c3" -->xlmr,  "17g3"-->xlmv
    "message": " |d all big with class weights",
    "dummy_run": False,
    "rseed": 123,
    "nw": 0,
    "max_len": 512,
    "train_ratio": 0.7,
    "val_ratio": 0.2,
    "test_ratio": 0.1,
    "epochs": 1,
    "encoder_alpha": 1e-05,
    "task_alpha": 0.0003,
    "snapshot_path": "data/",
    "threshold": 0.5,
    "include_prev_context": True,
    "include_title": False,
    "datasize": "big",
    "loss": "wfl",
    "pos_embeddings": False,
    "posE_V": 4181,
    "posE_D": 20,
    "comparison": "para-level",
    "fl_gamma": 2.0,
    "save_every_x_iter": 5000,
    "batch_size_list": "[12, 12, 12, 12, 12]",
    "world_size": 5
}

hyper_params['bert_model_name'] = 'xlm-roberta-base' if '17c3' in hyper_params['run_ID'] else 'facebook/xlm-v-base'

signature = {
    'max_len':hyper_params['max_len'],
    'bert_model_name':hyper_params['bert_model_name'],
    'include_prev_context':hyper_params['include_prev_context'],
    'include_title':hyper_params['include_title'],
    'datasize':hyper_params['datasize'],
    'pos_embeddings':hyper_params['pos_embeddings'],
    'comparison':hyper_params['comparison'],
    'message':'all big with class weights'
}

os.environ['PYTHONHASHSEED'] = str(hyper_params['rseed'])
# Torch RNG
torch.manual_seed(hyper_params['rseed'])
torch.cuda.manual_seed(hyper_params['rseed'])
torch.cuda.manual_seed_all(hyper_params['rseed'])
# Python RNG
np.random.seed(hyper_params['rseed'])
random.seed(hyper_params['rseed'])

class model_class(torch.nn.Module):
    def __init__(self,hyper_params, class_weights):
        super(model_class, self).__init__()
        self.hyper_params = hyper_params
        self.trainable = {}
        model_path = 'data/model_cache/'
        if 'google/mt5' in self.hyper_params['bert_model_name']:
            self.encoder = MT5EncoderModel.from_pretrained(self.hyper_params['bert_model_name'], return_dict=True, output_hidden_states=True, cache_dir=model_path)
        else:
            self.encoder = AutoModel.from_pretrained(self.hyper_params['bert_model_name'], return_dict=True, output_hidden_states=True, cache_dir=model_path)
        self.enc_config = AutoConfig.from_pretrained(self.hyper_params['bert_model_name'])
        self.tokenizer = AutoTokenizer.from_pretrained(self.hyper_params['bert_model_name'])
        self.trainable['encoder'] = self.encoder
        self.class_weights = class_weights
        if hyper_params['pos_embeddings']:
            self.pos_embeddings = nn.Embedding(hyper_params['posE_V']+2,hyper_params['posE_D'])
            self.trainable['pos_embeddings'] = self.pos_embeddings
            poslen = 20
        else:
            poslen = 0
        self.fine_tuning_layers = nn.Sequential(nn.Linear(self.enc_config.hidden_size+poslen,100),nn.ReLU(),nn.Dropout(0.2),nn.Linear(100,50),nn.ReLU(),nn.Dropout(0.2),nn.Linear(50,1),nn.Sigmoid())
        self.trainable['fine_tuning_layers'] = self.fine_tuning_layers
        self.gamma = hyper_params['fl_gamma']
        if hyper_params['loss'] == 'bce':
            self.loss_fun = nn.BCELoss()
        elif hyper_params['loss'] == 'wbce':
            self.loss_fun = self.BCELoss_ClassWeights
        elif hyper_params['loss'] == 'fl':
            self.loss_fun = self.FocLoss
        elif hyper_params['loss'] == 'wfl':
            self.loss_fun = self.FocLoss_ClassWeights
        else:
            raise 'error'
        self.best_val_metric = 0
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
    
    def BCELoss_ClassWeights(self, logits, labels):
        class_weights = self.class_weights
        logits = torch.clamp(logits,min=1e-7,max=1-1e-7)
        bce = - ( class_weights[1] * labels * torch.log(logits) + class_weights[0] * (1 - labels) * torch.log(1 - logits) )
        return torch.mean(bce)
    
    def FocLoss_ClassWeights(self, logits, labels):
        class_weights = self.class_weights
        logits = torch.clamp(logits,min=1e-7,max=1-1e-7)
        bce = - ( class_weights[1] * labels * ((1-logits)**self.gamma) * torch.log(logits) + class_weights[0] * (1 - labels) * (logits**self.gamma) * torch.log(1 - logits) )
        return torch.mean(bce)
    
    def FocLoss(self, logits, labels):
        logits = torch.clamp(logits,min=1e-7,max=1-1e-7)
        bce = - ( labels * ((1-logits)**self.gamma) * torch.log(logits) + (1 - labels) * (logits**self.gamma) * torch.log(1 - logits) )
        return torch.mean(bce)

    def forward(self, input_ids, attention_mask, pos, labels=None):
        bert_output = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:,0] # embedding of cls token in every batch
        if hyper_params['pos_embeddings']:
            pos_e = self.pos_embeddings(pos)
            out = torch.concat([bert_output, pos_e],dim=1)
        else:
            out = bert_output
        logits = self.fine_tuning_layers(out).view(-1)
        if labels is not None:
            assert logits.shape == labels.shape
            # print(logits.type(), labels.type())
            # input('wait')
            loss = self.loss_fun(logits, labels)
        else:
            loss = None
        return SequenceClassifierOutput(loss=loss, logits=logits)
    
    def predict(self, context, question):
        context_lines = context.split('\n')
        text, pos = [], []
        for i,_ in enumerate(context_lines):
            a_part = ' '.join(context_lines[:i+1])
            len_q_part = len(self.tokenizer.tokenize(question))
            a_part = self.tokenizer.convert_tokens_to_string(self.tokenizer.tokenize(a_part)[-(self.hyper_params['max_len']-len_q_part):])
            text.append(question+' '+a_part)
            pos.append(i)
        
        # tok = self.tokenizer(text, padding=True, truncation=True, max_length=self.hyper_params['max_len'], return_tensors='pt')
        # input_ids = tok['input_ids'].to(self.device)
        # attention_mask = tok['attention_mask'].to(self.device)
        # pos = torch.tensor(pos).to(self.device)
        # with torch.no_grad():
        #     logits = self.forward(input_ids, attention_mask, pos).logits
        logits = self.predict_this(text)
        # result = []
        # for line, score in zip(context_lines, logits.tolist()):
        #     result.append((line, score))
        return logits.tolist()

    def predict_this(self, text):
        logits_list = []
        offset = 8 # # reduce for small GPUs
        i = 0
        while i < len(text):
            tok = self.tokenizer(text[i:i+offset], padding=True, truncation=True, max_length=self.hyper_params['max_len'], return_tensors='pt')
            input_ids = tok['input_ids'].to(self.device)
            attention_mask = tok['attention_mask'].to(self.device)
            with torch.no_grad():
                logits = self.forward(input_ids, attention_mask, None).logits
            logits_list.append(logits.detach().cpu())
            i+=offset
        logits = torch.concat(logits_list, dim=0)
        return logits

if __name__ == '__main__':
    model = model_class(hyper_params, [])
    model = model.to(model.device)
    model.eval()

    print('loading model',hyper_params['run_ID'])
    snapshot = torch.load(hyper_params['snapshot_path']+'snapshot'+hyper_params['run_ID']+'.pt', map_location="cuda:0")
    print(model.load_state_dict(snapshot["MODEL_STATE"]))
    
    # https://www.bbc.com/news/uk-scotland-scotland-business-61908804 
    context = "Published24 June 2022\nA £2m fund aimed at reducing the environmental impact of textiles has been launched in Scotland.\nZero Waste Scotland (ZWS) said the money would go directly to textile businesses across the nation, from fashion to upholstery.\nIt estimates that while textiles make up just 4% of waste by weight, they account for 32% of the carbon impact of Scotland's household waste.\nThe Circular Textiles Fund is designed to boost Scotland's circular economy.\nIt is backed by the Scottish government and will be administered in three rounds as grant funding.\nZWS said it was supporting initiatives that would cut demand for new clothing, employ sustainable manufacturing processes, mitigate the pollution from washing textiles and make them easier to reuse and repair.\nThe organisation also wants to \"maximise the potential\" of waste textiles.\nIt is keen on initiatives that include material sorting technologies to better sort and grade textiles for reuse, the creation of new products from waste and the reprocessing of problem materials.\nThe circular economy is a model of production and consumption which involves sharing, leasing, reusing, repairing, refurbishing and recycling existing materials and products for as long as possible.\nIn practice, it implies reducing waste to a minimum. When a product reaches the end of its life, its materials are kept within the economy wherever possible. These can be productively used again and again, thereby creating further value.\nSource: European Parliament\nZero Waste Scotland chief executive Iain Gulland said: \"As a nation, we need to rethink the way we make, buy, and use products and take action to consume more responsibly.\n\"Businesses have a key role to play in facilitating that shift, helping customers make more sustainable purchasing decisions while also contributing to a greener economy.\n\"With textiles responsible for such a significant chunk of the carbon footprint of Scotland's household waste, it's vital that we move away from a throwaway approach to products and materials and make things last instead.\"\nThe funding announcement comes shortly after the Scottish government launched two public consultations on proposals for a Circular Economy Bill and Route Map to 2025.\nCircular Economy Minister Lorna Slater said the fund would help businesses in Scotland \"turn their proposals into reality\".\nShe said: \"Every material that is wasted comes at a cost to our planet, but it's clear that textiles are having a disproportionate environmental impact.\n\"From fashion to furniture, there are huge opportunities for businesses with creative ideas to help address that problem.\"\nThe initiative has been welcomed by companies already contributing to the circular economy.\nAndrew Rough, chief executive of Glasgow-based Advanced Clothing Solutions, which offers a clothing rental service to retailers, said: \"If we can get as much support as possible from the government, that will really help us to change the mindset and get people to think about rental and resale.\n\"We still rent out some kilts that we have had since 1997, so renting is really good from an economic perspective.\n\"We've got some items that we've rented out 50 to 100 times. When you think about that compared to a traditional model, it is very profitable for our customers.\""
    context = "The organisation also wants to \"maximise the potential\" of waste textiles.\nIt is keen on initiatives that include material sorting technologies to better sort and grade textiles for reuse, the creation of new products from waste and the reprocessing of problem materials.\nThe circular economy is a model of production and consumption which involves sharing, leasing, reusing, repairing, refurbishing and recycling existing materials and products for as long as possible.\nIn practice, it implies reducing waste to a minimum. When a product reaches the end of its life, its materials are kept within the economy wherever possible. These can be productively used again and again, thereby creating further value.\nZero Waste Scotland chief executive Iain Gulland said: \"As a nation, we need to rethink the way we make, buy, and use products and take action to consume more responsibly.\n\"Businesses have a key role to play in facilitating that shift, helping customers make more sustainable purchasing decisions while also contributing to a greener economy."
    question = "What is the circular economy?"

    print('Preparing input')
    text_list = []
    tokenizer = model.tokenizer
    for para_i,para in tqdm(enumerate(context.split('\n')), total=len(context.split('\n')), ncols=100):
        # including previous para till the token limit is reached
        # the code for that is as follows
        len_q_part = len(tokenizer.tokenize(question))
        a_part = '\n'.join(context.split('\n')[:para_i+1])
        a_part = tokenizer.convert_tokens_to_string(tokenizer.tokenize(a_part)[-(model.hyper_params['max_len']-len_q_part):])
        target_text = question+' '+a_part
        text_list.append(target_text)
            
    print('Running inference on',len(text_list),'paragraphs')
    a = time.time()
    logits = model.predict_this(text_list)
    print(logits)
    print('Time taken:',time.time()-a,'seconds')
    input('ENTER to exit')

    