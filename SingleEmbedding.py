# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 23:16:59 2022

@author: Bestom
"""

import json
import cv2
import numpy as np
import torch
from transformers import BertTokenizer, BertModel

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
#logging.basicConfig(level=logging.INFO)

import matplotlib.pyplot as plt
#%matplotlib inline

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


# Load pre-trained model (weights)
model = BertModel.from_pretrained('bert-base-uncased',
                                  output_hidden_states = True, # Whether the model returns all hidden-states.
                                  )

# Put the model in "evaluation" mode, meaning feed-forward operation.
model.eval()


import json

import time

from nltk.stem import WordNetLemmatizer
#莫名其妙要下载
#import nltk
#nltk.download('wordnet') 
#nltk.download('omw-1.4')
lemmatizer = WordNetLemmatizer()


def get_single_embedding(offer_text):

    sample_text=offer_text
    phrase_list={}
    map_exist=0
    
    

    # Add the special tokens.
    marked_text = "[CLS] " + sample_text + " [SEP]"
    
    # Split the sentence into tokens.
    tokenized_text = tokenizer.tokenize(marked_text)
    
    # Map the token strings to their vocabulary indeces.
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        
        
    # Mark each of the 22 tokens as belonging to sentence "1".
    segments_ids = [1] * len(tokenized_text)
    
    
    #3333333333333333333333333
    
    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    
    
    
    
    # Run the text through BERT, and collect all of the hidden states produced
    # from all 12 layers. 
    with torch.no_grad():
    
        outputs = model(tokens_tensor, segments_tensors)
    
        # Evaluating the model will return a different number of objects based on 
        # how it's  configured in the `from_pretrained` call earlier. In this case, 
        # becase we set `output_hidden_states = True`, the third item will be the 
        # hidden states from all layers. See the documentation for more details:
        # https://huggingface.co/transformers/model_doc/bert.html#bertmodel
        hidden_states = outputs[2]
        

        
        
    # Concatenate the tensors for all layers. We use `stack` here to
    # create a new dimension in the tensor.
    token_embeddings = torch.stack(hidden_states, dim=0)
    
    #print(token_embeddings.size())
        
    # Remove dimension 1, the "batches".
    token_embeddings = torch.squeeze(token_embeddings, dim=1)
    
    #print(token_embeddings.size())
    
    # Swap dimensions 0 and 1.
    token_embeddings = token_embeddings.permute(1,0,2)
    
    #print(token_embeddings.size())
    #3.3
    # Stores the token vectors, with shape [22 x 3,072]
    token_vecs_cat = []
    
    # `token_embeddings` is a [22 x 12 x 768] tensor.
    
    # For each token in the sentence...
    for token in token_embeddings:
        
        # `token` is a [12 x 768] tensor
    
        # Concatenate the vectors (that is, append them together) from the last 
        # four layers.
        # Each layer vector is 768 values, so `cat_vec` is length 3,072.
        cat_vec = torch.cat((token[-1], token[-2], token[-3], token[-4]), dim=0)
        
        # Use `cat_vec` to represent `token`.
        token_vecs_cat.append(cat_vec)
    
    #print ('Shape is: %d x %d' % (len(token_vecs_cat), len(token_vecs_cat[0])))    
        
        
    # Stores the token vectors, with shape [22 x 768]
    token_vecs_sum = []
    
    # `token_embeddings` is a [22 x 12 x 768] tensor.
    
    # For each token in the sentence...
    for token in token_embeddings:
    
        # `token` is a [12 x 768] tensor
    
        # Sum the vectors from the last four layers.
        sum_vec = torch.sum(token[-4:], dim=0)
        
        # Use `sum_vec` to represent `token`.
        token_vecs_sum.append(sum_vec)
    
    #print ('Shape is: %d x %d' % (len(token_vecs_sum), len(token_vecs_sum[0])))
    
    # `hidden_states` has shape [13 x 1 x 22 x 768]
    
    # `token_vecs` is a tensor with shape [22 x 768]
    token_vecs = hidden_states[-2][0]
    
    # Calculate the average of all 22 token vectors.
    sample_sentence_embedding = torch.mean(token_vecs, dim=0)



    sample_sentence_embedding=np.abs(sample_sentence_embedding)
    a=torch.zeros(1,784-768)
    print(a)
    sample_sentence_embedding=torch.cat((sample_sentence_embedding,a[0]),0)
    sample_sentence_embedding=sample_sentence_embedding.reshape(28,28)

    
    sample_sentence_embedding = np.array(sample_sentence_embedding).reshape(28,28)
    # resize
    # sample_sentence_embedding = cv2.resize(sample_sentence_embedding, (28, 28))       
    # print(sample_sentence_embedding)
    
    return sample_sentence_embedding
