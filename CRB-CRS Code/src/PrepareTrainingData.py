#Importing the modules
import pandas as pd
import numpy as np
import os
import simplejson as json

class PrepareTrainingData:
    def __init__(self):
        self.redial_data = None
        self.PATH = os.path.dirname(os.path.abspath(__file__))
        self.ROOT_DIR_PATH = os.path.abspath(os.path.dirname(self.PATH))
        self.DATA_path = os.path.join(self.ROOT_DIR_PATH, 'data\\dialog_data\\')
        print('object created')

    def read_input_json_file(self,filename):
        with open(filename, 'r', encoding='utf-8') as json_file:
            self.data = json.load(json_file)

    def parse_dialogues(self, data):
        dialogs = data['foo']
        text_messages_raw = []
        counter =0
        for key, d in enumerate(dialogs):
            #text_messages_raw = []
            messages = dialogs[key]['messages']
            seeker_id = dialogs[key]['initiatorWorkerId']
            recommender_id = dialogs[key]['respondentWorkerId']
            seeker_text = ''
            gt_text = ''
            counter = counter +1
            text_messages_raw.append('CONVERSATION:'+ str(counter))
            for msgid, msg in enumerate(messages):

                senderId = messages[msgid]['senderWorkerId']
                if senderId == seeker_id:
                    if gt_text:
                        text_messages_raw.append('GT~' + gt_text)
                        gt_text = ''
                        seeker_text =  seeker_text +' '+ messages[msgid]['text']
                    else:
                        seeker_text =  seeker_text +' ' + messages[msgid]['text']

                elif senderId == recommender_id:
                    if seeker_text:
                        text_messages_raw.append('SKR~' + seeker_text)
                        seeker_text = ''
                        gt_text = gt_text+' '  + messages[msgid]['text']
                    else:
                        gt_text = gt_text +' ' + messages[msgid]['text']

            if gt_text:
                text_messages_raw.append('GT~' + gt_text)
            elif seeker_text:
                text_messages_raw.append('SKR~' + seeker_text)
        return text_messages_raw

    def write_data(self, raw_messages_data):
        with open(self.DATA_path+'TrainingDataParsed_Con.txt', 'w', encoding='utf-8') as filehandle:
            for line in raw_messages_data:
                filehandle.write("%s\n" % line)


if __name__ == '__main__':
    obj= PrepareTrainingData()
    filename = os.path.join(obj.DATA_path+'Unparsed_train_data.txt')
    obj.read_input_json_file(filename)
    raw_messages_data = obj.parse_dialogues(obj.data)
    obj.write_data(raw_messages_data)
    print('data exported')
