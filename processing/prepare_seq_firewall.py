import datetime
import random
import re
from typing import List, Text

import pandas as pd
import numpy as np

from tqdm import tqdm

from transformers import AutoTokenizer

''' '''

class PreprocessFirewall(object):
    def __init__(self, logs: List[Text]) -> None:
        self.logs = logs

    @staticmethod
    def _cleanTimelineMessage(l: Text):
        l = re.sub(r'^.*?%ASA-\w+-\d-\d+:', '', l)
        # OR
        l = re.sub(r'^.*?%ASA--\d-\d+:', '', l) 
        # OR 
        #l = re.sub(r'^.*?%ASA--\d-\d+:', '', l) 
        # %ASA-bridge-6-1100
        # there are some messages 
        # started with %ASA--4-733:
        # started with %ASA-session-\d-\d+::
        # manually omitted
        return l
   
    @staticmethod
    def _cleanParanthesis(l: Text):
        '''for cleaning extra IP information '''          
        l = re.sub(r'\(([^()]*)\)', '', l)
        return l
    
    @staticmethod
    def _info_bracket_fix(l: Text):
        ''' clean bracket around information '''
        xxx_match = [xxx.group() for xxx in re.finditer(r"(\[)[a-z ]+(\])",l)]
        xxx_bound = [xxx.groups() for xxx in re.finditer(r"(\[)[a-z ]+(\])",l)]
        xxx_out = l
        if len(xxx_match)>0:                     
            for xxxx in xxx_bound[0]:
                xxx_out = xxx_out.replace(xxxx,"")
        return xxx_out 

    @staticmethod
    def _clean_HEX(l: Text):
        xxx = re.sub(r"((?<=[^A-Za-z0-9])|^)(0x[a-f0-9A-F]+)((?=[^A-Za-z0-9])|$)","",l)
        xxx = re.sub("\[, \]","",xxx)
        return xxx
    
    @staticmethod
    def _augment_some_special_chars(l: Text):
        xx = re.sub("\B_\B"," ",l)
        xx = re.sub("->","to",xx)

        return xx

    @staticmethod
    def _cleanlefovers(l: Text):
        '''for cleaning extra brackets for hexadecimal number '''      
        l = re.sub(r'[\[\],]', '', l)
        return l

    @staticmethod
    def _fix_missing_IP(l: Text):
        ''' if there is x in it, attain some number to IT '''
        # ^ start of the line
        # $ end of the line
        REGEX_parts = r"(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?|[a-z])"
       
        R = re.compile(REGEX_parts, re.S)
        
        set_x = random.randrange(1, 256, 1)
        l_edited = R.sub(lambda m: m.group().replace('.x', str(set_x), 1), l)
        return l_edited

    @staticmethod
    def _fix_range_IP(l: Text):
        ''' if there is range in it, attain some number to IT '''
        # ^ start of the line
        # $ end of the line
        REGEX_parts = r"(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.((\d|\d\d+|1\d\d+|2[0-4]\d)\-(1\d\d*|2[0-4]\d|250))"
       
        R = re.compile(REGEX_parts, re.S)
        
        l_edited = R.sub(lambda m: m.group().replace(
            m[4], str(random.randrange(int(m[5]), int(m[6]), 1)), 1), l)
        return l_edited

    @staticmethod     
    def _fix_emptystrings(l: Text):
        ''' removes extra empty lines '''       

        xx = re.sub(r"^[ \t\r\n]+|[ \t\r\n]+$","",l)
        xx = re.sub(r"\s{2,}"," ",l)      
        return xx

    #@staticmethod
    #def _getTime(l: Text):
    #    ''' extracts time from log line '''
    #    found = re.findall(
    #        "((00|[0-9]|1[0-9]|2[0-3]):([0-9]*|[0-5][0-9]):([0-9]*|[0-5][0-9]))", l.strip())
    #    foundTime = datetime.time(
    #        int(found[0][1]), int(found[0][2]), int(found[0][3]))
    #    return foundTime 

    def clean(self, l: Text):
        # line edited - le
        le = self._cleanTimelineMessage(l)
        le = self._cleanParanthesis(le)   
        le = le.lower()                 
        le = self._clean_HEX(le)
        le = self._info_bracket_fix(le)  
        le = self._augment_some_special_chars(le)        
        le = self._fix_missing_IP(le)
        le = self._fix_range_IP(le)
        le = self._fix_emptystrings(le)        
        
        return le

    def save(self):
        ''' save log file as csv '''
        try:
            import pandas as pd
            from tqdm import tqdm
        except ImportError:
            raise

        print('Saving the file...')
        df = {'time': [], 'log': []}
        # For every sentence...
        tn = self.__len__()  # total number of logs
        print(f'Total number of logs: {tn}')
        stride = 1  # a step for tqdm

        # checking up file through a file viewer is important so we limit row length with 1M
        LIMIT = 1000000
        startTime = datetime.datetime.now().strftime("%H_%M_%S")

        parts = int(tn/LIMIT)
        residual = tn % LIMIT
        parts = parts+1 if residual > 0 else parts

        for part in range(0, parts, stride):
            df = {'time': [], 'log': []}
            start = part*LIMIT
            end = start+LIMIT if tn-start > LIMIT else start+residual
            print(f'Part-{part+1} working...')
            for i in tqdm(range(start, end, stride)):
                line = self.logs[i]
                # if there is a match
                foundTime = self._getTime(line)
                # line edited - le
                le = self.clean(line)
                df['log'].append(le)
                df['time'].append(foundTime)

            df = pd.DataFrame(data=df, columns=['time', 'log'])
            df.to_csv("data/firewall/anomaly-log-part-" + str(part+1) + "-" +
                      startTime+".csv", index=False, header=True)
            del(df)

    def __getitem__(self, idx):
        #time = self._getTime(self.logs[idx])
        text = self.clean(self.logs[idx])

        #return (time, text)
        return text

    def __len__(self):
        return len(self.logs)

def main():
    #bids_logs = PreprocessFirewall(out['log'].tolist())

    # DAY1
    p1 = pd.read_csv("data/firewall/anomaly/day1-labeled-part1.csv", sep=',')
    p2 = pd.read_csv("data/firewall/anomaly/day1-labeled-part2.csv", sep=',')
    p3 = pd.read_csv("data/firewall/anomaly/day1-labeled-part3.csv", sep=',')
    p4 = pd.read_csv("data/firewall/anomaly/day1-labeled-part4.csv", sep=',')
    p5 = pd.read_csv("data/firewall/anomaly/day1-labeled-part5.csv", sep=',')
    p6 = pd.read_csv("data/firewall/anomaly/day1-labeled-part6.csv", sep=',')
    p7 = pd.read_csv("data/firewall/anomaly/day1-labeled-part7.csv", sep=',')
    p8 = pd.read_csv("data/firewall/anomaly/day1-labeled-part8.csv", sep=',')
    p9 = pd.read_csv("data/firewall/anomaly/day1-labeled-part9.csv", sep=',')
    p10 = pd.read_csv("data/firewall/anomaly/day1-labeled-part10.csv", sep=',')
    p11 = pd.read_csv("data/firewall/anomaly/day1-labeled-part11.csv", sep=',')
    p12 = pd.read_csv("data/firewall/anomaly/day1-labeled-part12.csv", sep=',')
    whole_day1 = [p1, p2, p3, p4, p5, p6, p7, p8, p9,p10,p11,p12]
    df_day1 = pd.concat(whole_day1)
    #df_day1.info()
    #df_day1['atype'] = np.where(df_day1.label == 1, 'Collective','-')     
    df_day1['type'] = np.where(df_day1.label == 1, 'DDOS','NORMAL')     


    # DAY 2
    p1 = pd.read_csv("data/firewall/anomaly/day2-labeled-part1.csv", sep=',')
    p2 = pd.read_csv("data/firewall/anomaly/day2-labeled-part2.csv", sep=',')
    p3 = pd.read_csv("data/firewall/anomaly/day2-labeled-part3.csv", sep=',')
    whole_day2 = [p1, p2, p3]
    df_day2 = pd.concat(whole_day2)

    #df_day2['atype'] = np.where(df_day2.log.str.contains(r"192.168.2.175/55892|192.168.2.175/55891|10.200.150.201"), 'Collective','-')#conditional     
    df_day2['type'] = np.where(df_day2.log.str.contains(r"192.168.2.175/55892|192.168.2.175/55891"), 'PS','NORMAL')
    df_day2.loc[df_day2.log.str.contains(r"10.200.150.201"), 'type'] = 'RD'
    #df_day2.loc[df_day2.log.str.contains(r"10.200.150.201"), 'atype'] = 'Point'    
    
    #df_day2.info()
    # DAY 3
    df_day3 = pd.read_csv("data/firewall/anomaly/day3-labeled.csv", sep=',')    
    df_day3['type'] = np.where(df_day3.log.str.contains("192.168.2.251"), 'RD','NORMAL')
    #df_day3['atype'] = np.where(df_day3.log.str.contains("192.168.2.251"), 'Point','-')
    
    # Concat data first
    days = [df_day1, df_day2, df_day3]
    whole_data = pd.concat(days)    
    # lets init tokenizer here
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    # get groups
    grouped = whole_data.groupby(["type"])
    types = ['DDOS','PS','RD','NORMAL']
    # new
    adict={'log':[],'label':[],'type':[],'time':[]}
    
    def add_row(arow,alabel,atype,atime):
        adict['log'].append(arow)
        adict['label'].append(alabel)
        adict['type'].append(atype)
        adict['time'].append(atime)

    for t in types:
        print(f'TYPE: {t} \n')

        # anomaly group
        if t != 'DDOS':
            if t == 'PS':                
                ag = grouped.get_group(t).sample(frac=0.1, random_state=666)
            else: 
                ag = grouped.get_group(t)   
        else:
            ag = grouped.get_group(t).sample(frac=0.01, random_state=666)

        # preproces and get all logs
        ag_logs = PreprocessFirewall(ag['log'].tolist())
        ag_time = ag['time'].tolist()
        # gel label
        ag_label = ag['label'].tolist()[0]

        #     
        arow = ""
        # expecting to create max or less 512 token length log and its label
        count=0         
        for i in tqdm(range(len(ag_logs))):
            alog = ag_logs[i]    
            atime = ag_time[i]        
            
            if t != 'RD':
                input_ids = tokenizer.encode(alog, add_special_tokens=False)
                count += len(input_ids) 
                if count <= 512: 
                    arow += alog +" "+tokenizer.sep_token+" "
                    if i == len(ag_logs)-1:
                        add_row(arow[:-1],ag_label,t,atime)                        

                        count = 0
                else:
                    add_row(arow,ag_label,t,atime)
                    # re-init
                    count = 0
                    arow = alog +" "+tokenizer.sep_token+" "
                    input_ids = tokenizer.encode(alog, add_special_tokens=False)
                    count += len(input_ids) 
            else:
                arow = alog +" "+tokenizer.sep_token
                add_row(arow,ag_label,t,atime)
    
    adf = pd.DataFrame(adict)
    print(adf.head(5))
    print("==========")
    print(adf.tail(5))
    adf.to_csv("data/firewall/anomaly/prepared.csv", index=False, header=True)
    # do stratified according to 'type'

def read():
    adf = pd.read_csv("data/firewall/anomaly/prepared.csv",sep=',') 
    print(adf.tail(2))
    #x = adf['log'].tolist()
    #print(f'First example: {x[0]} \n Last Example: {x[-1]}')
    print(len(adf))
    print(len(adf[adf['label'] == 0]))
    

if __name__ == "__main__":
    read()