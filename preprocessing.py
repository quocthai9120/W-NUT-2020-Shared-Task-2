import pandas as pd

import torch
import argparse

from transformers import RobertaConfig
from transformers import RobertaModel

from fairseq.data.encoders.fastbpe import fastBPE
from fairseq.data import Dictionary

absolute_path = "/home/ubuntu/BERT-baseline/"

# Load model
config = RobertaConfig.from_pretrained(
    "{}/BERTweet_base_transformers/config.json".format(absolute_path)
)
BERTweet = RobertaModel.from_pretrained(
    "{}/BERTweet_base_transformers/model.bin".format(absolute_path),
    config=config
)

# Load BPE encoder 
parser = argparse.ArgumentParser()
parser.add_argument('--bpe-codes', 
    default="{}/BERTweet_base_transformers/bpe.codes".format(absolute_path),
    required=False,
    type=str,  
    help='path to fastBPE BPE'
)
args = parser.parse_args()
bpe = fastBPE(args)

# Load the dictionary  
vocab = Dictionary()
vocab.add_from_file("{}/BERTweet_base_transformers/dict.txt".format(absolute_path))

#------------------------------------
# INPUT TEXT IS TOKENIZED!
#line = "SC has first two presumptive cases of coronavirus , DHEC confirms HTTPURL via @USER :cry:" 
#-------------------------------------

# INPUT TEXT IS TOKENIZED!
line = "SC has first two presumptive cases of coronavirus , DHEC confirms HTTPURL via @USER :cry:" 

# Encode the line using fastBPE & Add prefix <s> and suffix </s> 
subwords = '<s> ' + bpe.encode(line) + ' </s>'

# Read dataset into pandas frame
df = pd.read_csv("./COVID19Tweet/train.tsv", sep='\t')

# Drop the id column
df = df.drop(['Id'], axis=1)
#print(df.head())

# Initialize a list
all_input_ids = torch.tensor([], dtype=torch.long)

# Overwrite embeddings onto orginal dataframe elements
#for i in range(len(df.index)):
#    temp = df.iloc[i, df.columns.get_loc('Text')] 
#    # Encode the line using fastBPE & Add prefix <s> and suffix </s> 
#    subwords = '<s> ' + bpe.encode(temp) + ' </s>'
    # Map subword tokens to corresponding indices in the dictionary
#    input_ids = vocab.encode_line(subwords, append_eos=False, add_if_not_exist=False).long().tolist()


# Convert into torch tensor
all_input_ids = torch.tensor([input_ids], dtype=torch.long)

# Extract features  
with torch.no_grad():  
    features = BERTweet(all_input_ids)  

# Represent each word by the contextualized embedding of its first subword token  
# i. Get indices of the first subword tokens of words in the input sentence 
listSWs = subwords.split()  
firstSWindices = []  
for ind in range(1, len(listSWs) - 1):  
    if not listSWs[ind - 1].endswith("@@"):  
        firstSWindices.append(ind)  

# ii. Extract the corresponding contextualized embeddings  
words = line.split()  
assert len(firstSWindices) == len(words)  
vectorSize = features[0][0, 0, :].size()[0]  
for word, index in zip(words, firstSWindices):  
    print(word + " --> " + " ".join([str(features[0][0, index, :][_ind].item()) for _ind in range(vectorSize)]))
    #print(type(word + " --> " + " ".join([str(features[0][0, index, :][_ind].item()) for _ind in range(vectorSize)])))
    
    







