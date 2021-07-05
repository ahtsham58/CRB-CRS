import sys
print(sys.path)
import pandas as pd
import numpy as np
import re
import os
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
class PreprocessSentences:
    def __init__(self):
        self.PATH = os.path.dirname(os.path.abspath(__file__))
        self.ROOT_DIR_PATH = os.path.abspath(os.path.dirname(self.PATH))
        self.DATA_path = os.path.join(self.ROOT_DIR_PATH, 'data\\dialog_data\\')
        print('object created')
    def seeker_sentences_parser(self, line):
        if line:
            p = re.compile("SEEKER:(.*)").search(str(line))
            temp_line = p.group(1)
            m = re.compile('<s>(.*?)</s>').search(temp_line)
            seeker_line = m.group(1)
            seeker_line = seeker_line.lower().strip()
            return seeker_line

    def gt_sentence_parser(self, line):
        try:
            if not line == '\n':
                p = re.compile("GROUND TRUTH:(.*)").search(str(line))
                temp_line = p.group(1)
                m = re.compile('<s>(.*?)</s>').search(temp_line)
                gt_line = m.group(1)
                gt_line = gt_line.lower().strip()
                # gt_line = re.sub('[^A-Za-z0-9]+', ' ', gt_line)
            else:
                gt_line = ""
        except AttributeError as err:
                print('exception accured while parsing ground truth.. \n')
                print(line)
                print(err)
                return gt_line

    #read and retrive dialogs from the input file
    def read_preprocess_sentences(self, file_name):
        redial_dialogues = []
        counter =0
        previous_line = ''
        counter = 0
        with open(file_name, 'r', encoding='utf-8') as input:
            for line in input:
                try:
                    #if line.__contains__('~') and line.__contains__('SKR~'):
                    if line:
                        if line.__contains__('CONVERSATION:'):
                            redial_dialogues.append(line.replace('\n',''))
                            continue
                        else:
                            previous_line = line
                            line = self.replace_movieIds_withPL(line)
                            line = line.split('~')[1].strip().lower()
                            line = self.convert_contractions(line)
                            line = re.sub('[^A-Za-z0-9]+', ' ', line)
                            line = line.replace('im','i am').strip()
                            line = self.remove_stopwords(line)
                            if len(line) < 1:
                                redial_dialogues.append('**')
                            else:
                                redial_dialogues.append(line)
                    else:
                        #print('not found')
                        #print(line)
                        #print('previous line is ...' +previous_line)
                        print('line issue')
                        counter = counter+1
                except:
                    print((previous_line))
                    print(line)
                    continue
        #print(counter)

        return redial_dialogues
    def remove_stopwords(self, line):
        text_tokens = word_tokenize(line)
        tokens_without_sw = [word for word in text_tokens if not word in stopwords.words()]
        filtered_sentence = (" ").join(tokens_without_sw)
        print(filtered_sentence)
        return filtered_sentence


    def convert_contractions(self, line):
        #line = "What's the best way to ensure this?"
        filename = os.path.join(self.ROOT_DIR_PATH, 'data\contractions.txt')
        contraction_dict = {}
        with open(filename) as f:
            for key_line in f:
               (key, val) = key_line.split(':')
               contraction_dict[key] = val
            for word in line.split():
                if word.lower() in contraction_dict:
                    line = line.replace(word, contraction_dict[word.lower()])
        return line

    def replace_movieIds_withPL(self , line):
        try:
            if "@" in line:
                ids = re.findall(r'@\S+', line)
                for id in ids:
                    line = line.replace(id,'movieid')
                    #id = re.sub('[^0-9@]+', 'movieid', id)
        except:
            lines.append(line)
            print('exception occured here')
        return line
        print('execution ends here')



if __name__ == '__main__':
    obj = PreprocessSentences()
    filename = os.path.join(obj.DATA_path, 'test_traningdata.txt')
    sentencesData = obj.read_preprocess_sentences(filename)
    with open(obj.DATA_path +'TrainingDataPLSW.txt', 'w', encoding='utf-8') as filehandle:
        for line in sentencesData:
            filehandle.write("%s\n" % line)
    print('Dialogs have been preprocessed successfully.')
