import pandas as pd
import csv
from tqdm import tqdm



import re
from typing import List, Text
import string


from transformers import  AutoTokenizer

''' .. '''

class PreprocessHDFS(object):
    def __init__(self, logs: List[Text]) -> None:
        self.logs = logs              

    @staticmethod
    def _cleanTimelineMessage(l: Text):
        l = re.sub(r'^.*?:', '', l)       
        return l 
    
    @staticmethod     
    def _fix_emptystrings(l: Text):
        ''' removes extra empty lines '''       

        xx = re.sub(r"^[ \t\r\n]+|[ \t\r\n]+$","",l)
        xx = re.sub(r"\s{2,}"," ",l)      
        return xx

    @staticmethod   
    def _remove_block(l: Text):
        ''' removes extra empty lines '''
        l = re.sub("BLOCK*","", l)
        return l          
    
    def _extra_clean(self, l: Text):        
        simple_tokenized = l.split()
        punc_removed_v0 = " ".join([i for i in simple_tokenized if i not in string.punctuation])

        ipex = r"\/(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)"
        x =re.finditer(ipex,punc_removed_v0)

        strObj = punc_removed_v0
        #[(m.start(0), m.end(0)) for m in re.finditer(pattern, string)]
        y = [m.start(0) for m in x]
        # Slice string to remove character at index(es) y
        for i, index in enumerate(y):
            if len(strObj) > index:
                strObj = strObj[0 : index : ] + strObj[index + 1 : :] 
            
            if i+1<len(y):
                y[i+1] -= 1

        return strObj  

    def _get_rid(self, l: Text):            
        l = re.sub(r'blk_-','blk', l)
        #OR
        l = re.sub(r'blk_','blk', l)       
        
        return l

    def clean(self, l: Text):
        # line edited - le
        le = self._cleanTimelineMessage(l) 
        le = self._remove_block(le)
        le = self._extra_clean(le)   
        le = self._get_rid(le)
        le = self._fix_emptystrings(le)
        
        
        return le.lower()   

    def __getitem__(self, idx):          
        text = self.clean(self.logs[idx])       

        return text

    def __len__(self):
        return len(self.logs)




def main():

    with open('data/hdfs/anomaly/anomaly_label.csv') as f:
        next(f)  # Skip the header
        reader = csv.reader(f, skipinitialspace=True)
        hdfs_label = dict(reader)

    hdfs_dataset = pd.read_csv("data/hdfs/anomaly/hdfs-log-key-extracted-drain.csv")
    # Add anomaly information
    hdfs_dataset['label'] = [0 if hdfs_label[bid] =='Normal' else 1 for bid in hdfs_dataset['blockId']]

    def getKeys(dict):
        return dict.keys()

    blockIDs = getKeys(hdfs_label)
    # lets init tokenizer here
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    # get groups
    grouped = hdfs_dataset.groupby(["blockId"])

    # new
    adict={'log':[],'label':[]}

    for bid in tqdm(blockIDs):
        out = grouped.get_group(bid)
        # we can save OR create seqs with tokenizer.
        #out.to_csv("data/hdfs/anomaly/"+bid+"csv", index=False, header=True)
        # still
        # get blockid's logs
        #bids_logs=out['log'].tolist()
        bids_logs = PreprocessHDFS(out['log'].tolist())
        # get current label
        bid_label = out['label'].tolist()[0]
        #     
        arow = ""
        # expecting to create max or less 512 token length log and its label
        count=0   
        for i in range(len(bids_logs)):
            #alog = log.strip()
            alog = bids_logs[i]
            input_ids = tokenizer.encode(alog, add_special_tokens=False)
            count += len(input_ids) 
            if count <= 512: 
                arow += alog +" "+tokenizer.sep_token+" "
            else:
                adict['log'].append(arow)
                adict['label'].append(bid_label)
                # re-init
                count = 0
                arow = alog +" "+tokenizer.sep_token+" "
                input_ids = tokenizer.encode(alog, add_special_tokens=False)
                count += len(input_ids)            
        
    adf = pd.DataFrame(adict)
    adf.to_csv("data/hdfs/anomaly/prepared.csv", index=False, header=True)



if __name__ == "__main__":
    main()