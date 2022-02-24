from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import scipy.stats
import pandas as pd
import numpy as np
import nlp
import re
import os
import math
import random
import nltk
from nltk.corpus import stopwords
import itertools
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.lm.preprocessing import padded_everygram_pipeline
from scipy import spatial
from sent2vec.vectorizer import Vectorizer
from nltk.lm import MLE
import dill as pickle
from nltk.util import ngrams
from collections import Counter
from src.Cacululate_MLE_Probs import Cacululate_MLE_Probs

class RetrieveCandidateResponses_TFIDF:
    def __init__(self):
        self.PATH = os.path.dirname(os.path.abspath(__file__))
        self.ROOT_DIR_PATH = os.path.abspath(os.path.dirname(self.PATH))
        self.DATA_path = os.path.join(self.ROOT_DIR_PATH, 'data')
        self.DATA_path = os.path.join(self.DATA_path, 'dialog_data', "")
        self.corpus_dataSW = []
        self.corpus_data = []
        self.original_corpus = []
        self.input_dialogs = []
        self.generated_dialogs = []
        self.corpus_tfidfSW =None
        self.corpus_tfidf =None
        self.stored_sentences = None
        self.stored_embeddings = None
        self.vectorizerSW = None
        self.vectorizer = None
        self.error =  None
        self.MLE= None
        self.pref_keywords = ['scary','horror','pixar','graphic', 'classic','comedy', 'kids','funny', 'disney','comedies','action','family','adventure','crime','fantasy','thriller','scifi', 'documentary', 'science fiction', 'drama','romance','romances','romantic','mystery','mysteries','history','no preference','suspense' ]
        with open(self.DATA_path+'ParsedData_PLSW.txt',encoding='utf-8') as f:
            self.corpus_dataSW = f.read().splitlines()
        with open(self.DATA_path+'ParsedData_PL.txt',encoding='utf-8') as f:
            self.corpus_data = f.read().splitlines()
        with open(self.DATA_path+'TrainingDataParsed_Con.txt',encoding='utf-8') as f:
            self.original_corpus = f.read().splitlines()
        with open(self.DATA_path+'GT_corpus_tokens.txt',encoding='utf-8') as f:
            self.GT_corpus = f.read().splitlines()
        if len(self.input_dialogs) < 1:
             with open(self.DATA_path+'test.txt',encoding='utf-8') as f:
                 self.input_dialogs = f.read().splitlines()
             self.input_dialogs_for_history = self.read_input_dialogs(self.DATA_path + 'test.txt')
        self.vectorizerSW = TfidfVectorizer()
        self.vectorizer = TfidfVectorizer()
        self.corpus_tfidfSW = self.vectorizerSW.fit_transform(self.corpus_dataSW)
        self.corpus_tfidf = self.vectorizer.fit_transform(self.corpus_data)

    def seeker_sentences_parser(self, line):
        if line:
            p = re.compile("SEEKER:(.*)").search(str(line))
            temp_line = p.group(1)
            m = re.compile('<s>(.*?)</s>').search(temp_line)
            seeker_line = m.group(1)
            seeker_line = seeker_line.lower().strip()
            return seeker_line

    def read_input_dialogs(self, path_to_input_file):
            is_visited = False
            #print(path_to_input_file)
            dialogues = []
            dialog = []
            with open(path_to_input_file, 'r', encoding='utf-8') as input:
                for line in input:
                    if not line.strip(): continue
                    if 'CONVERSATION:' in line and is_visited:
                        dialogues.append(dialog)
                        dialog = []
                        dialog.append(line)
                        is_visited = False
                    else:
                        dialog.append(line)
                        is_visited = True
            dialogues.append(dialog)
            return dialogues

    def preprocess_sentences(self, line):
        gt_line = self.replace_movieIds_withPL(line)
        #gt_line = gt_line.split('~')[1].strip().lower()
        gt_line = self.convert_contractions(gt_line)
        gt_line = re.sub('[^A-Za-z0-9]+', ' ', gt_line)
        gt_line = gt_line.replace('im', 'i am').strip()
        return gt_line

    def convert_contractions(self, line):
        #line = "What's the best way to ensure this?"
        filename = os.path.join(self.DATA_path+'//contractions.txt')
        contraction_dict = {}
        with open(filename) as f:
            for key_line in f:
               (key, val) = key_line.split(':')
               contraction_dict[key] = val
            for word in line.split():
                if word.lower() in contraction_dict:
                    line = line.replace(word, contraction_dict[word.lower()])
        return line

    def remove_stopwords(self, line):
        text_tokens = word_tokenize(line)
        tokens_without_sw = [word for word in text_tokens if not word in stopwords.words()]
        filtered_sentence = (" ").join(tokens_without_sw)
        #print(filtered_sentence)
        return filtered_sentence

    #funtion to retrive candidates based on last seeker query
    def retrieve_candidates_by_last_seeker_query(self, query, nb_candidates):
        #Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
        top_k = 500
        counter = 0
        GT_candidate_list = []
        if query.__contains__('SEEKER:'):
            parsed_sentence = self.seeker_sentences_parser(query)
            preproc_sentence = self.preprocess_sentences(parsed_sentence)
            seeker_sentence = self.remove_stopwords(preproc_sentence)
            tokens = word_tokenize(seeker_sentence)
            if len(tokens) > 2:
                query_tfidf = self.vectorizerSW.transform([seeker_sentence])
                cosine_matrix = cosine_similarity(query_tfidf, self.corpus_tfidfSW).flatten()
            #similarity check for tokens less than 2
            else:
                seeker_sentence = preproc_sentence
                query_tfidf = self.vectorizer.transform([seeker_sentence])
                cosine_matrix = cosine_similarity(query_tfidf, self.corpus_tfidf).flatten()

            sim_sent_indices = cosine_matrix.argsort()[:-top_k:-1]
            sim_sent_scores = cosine_matrix[sim_sent_indices]

            for score, idx in zip(sim_sent_scores, sim_sent_indices):
                rt_corres_token_count =0
                rt_sentence = self.original_corpus[idx]
                if idx < len(self.original_corpus)-1:
                    try:
                        rt_corres_sentence = self.original_corpus[idx+1]
                        rt_corres_sentence = re.sub('[^A-Za-z0-9~]+', ' ', rt_corres_sentence)
                        text_tokens = word_tokenize(rt_corres_sentence.split('~')[1].strip())
                        rt_corres_token_count = len(text_tokens)
                    except  IndexError as err:
                        continue
                else:
                    rt_corres_sentence = rt_sentence
                if not rt_corres_sentence.__contains__('GT~') or rt_corres_token_count <= 3 or rt_corres_token_count > 20 :
                    continue
                elif not rt_sentence.__contains__('SKR~'):
                    continue
                else:
                    counter = counter+1
                    GT_candidate_list.append(str(self.original_corpus[idx+1]))
                    if counter == nb_candidates:
                        break
        return GT_candidate_list, tokens

    def gt_sentence_parser(self, line):
        try:
            if not line == '\n':
                p = re.compile("GROUND TRUTH:(.*)").search(str(line))
                temp_line = p.group(1)
                m = re.compile('<s>(.*?)</s>').search(temp_line)
                gt_line = m.group(1)
                gt_line = gt_line.strip().lower()
                return gt_line
                # gt_line = re.sub('[^A-Za-z0-9]+', ' ', gt_line)
            else:
                gt_line = ""
        except AttributeError as err:
                print('exception accured while parsing ground truth.. \n')
                #print(line)
                print(err)
                return gt_line

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

    #funtion to retrive candidates based on dialog history up to k, k is the parameter value represeting last k utterances as part
    #of user query
    def retrieve_sentences_history(self, dialog_history, nb_candidates):
        #Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
        counter = 0
        GT_candidate_list = []
        final_query = ''
        if dialog_history[len(dialog_history)-1].__contains__('SEEKER:'):
            query = ''
            for q in dialog_history:
                if q.__contains__('SEEKER:'):
                    final_query = final_query + str(self.seeker_sentences_parser(q))+','
                else:
                    final_query = final_query + str(self.gt_sentence_parser(q))+','

            final_query = self.preprocess_sentences(final_query)
            final_query = final_query.replace('None,','')
            query_tfidf = self.vectorizerSW.transform([final_query])
            cosine_matrix = cosine_similarity(query_tfidf, self.corpus_tfidfSW).flatten()
        top_k = 500
        sim_sent_indices = cosine_matrix.argsort()[:-top_k:-1]
        sim_sent_scores = cosine_matrix[sim_sent_indices]
        for score, idx in zip(sim_sent_scores, sim_sent_indices):
            rt_corres_token_count =0
            rt_sentence = self.original_corpus[idx]
            if idx < len(self.original_corpus)-1:
                try:
                    rt_corres_sentence = self.original_corpus[idx+1]
                    rt_corres_sentence = re.sub('[^A-Za-z0-9~]+', ' ', rt_corres_sentence)
                    text_tokens = word_tokenize(rt_corres_sentence.split('~')[1].strip())
                    rt_corres_token_count = len(text_tokens)
                except  IndexError as err:
                    continue
            else:
                rt_corres_sentence = rt_sentence
            if not rt_corres_sentence.__contains__('GT~') or rt_corres_token_count <= 3 or rt_corres_token_count > 20 :
                continue
            elif not rt_sentence.__contains__('SKR~'):
                continue
            else:
                counter =counter+1
                GT_candidate_list.append(str(self.original_corpus[idx+1]))
                if counter == nb_candidates:
                    break
        return GT_candidate_list

    # this function is the main entry point to retrive candidates, filter outliers, and rank valid candidates
    def process_dialogs(self, nb_candidates=5):
        generated_dialogs = []
        for dlg in self.input_dialogs_for_history:
            for index, input_query,  in enumerate(dlg):
                candidate_list_K1= []
                candidate_list_K2= []
                candidate_list_K3= []
                candidate_list_K4= []
                valid_pair_list_1= []
                valid_pair_list_2= []
                valid_pair_list_3= []
                valid_pair_list_4= []
                dialog = dlg[1:]
                generated_dialogs.append(input_query.replace('\n', ''))
                if input_query.__contains__('CONVERSATION:'):
                    continue
                elif input_query.__contains__('SEEKER:'):
                    if index ==1:
                        candidate_list_K1, seeker_tokens = self.retrieve_candidates_by_last_seeker_query(input_query, nb_candidates)
                        valid_pair_list_1 = self.filter_outliers_from_set(candidate_list_K1, nb_candidates)
                    else:
                        candidate_list_K1, seeker_tokens = self.retrieve_candidates_by_last_seeker_query(input_query, nb_candidates)
                        valid_pair_list_1 = self.filter_outliers_from_set(candidate_list_K1, nb_candidates)
                        query_2 = dialog[index-2:index]
                        candidate_list_K2 = self.retrieve_sentences_history(query_2, nb_candidates)
                        valid_pair_list_2 = self.filter_outliers_from_set(candidate_list_K2, nb_candidates)
                        query_3 = dialog[index-3:index]
                        if len(query_3) == 0: ## when dialog starts with ground-truth sentence,
                            query_3 = dialog[index-2:index]
                        candidate_list_K3 = self.retrieve_sentences_history(query_3, nb_candidates)
                        valid_pair_list_3 = self.filter_outliers_from_set(candidate_list_K3, nb_candidates)
                        complete_history = dlg[1:index+1]
                        candidate_list_K4 = self.retrieve_sentences_history(complete_history, nb_candidates)
                        valid_pair_list_4 = self.filter_outliers_from_set(candidate_list_K4, nb_candidates)

                    # combine candidates from all the valid filtered lists after discarding outliers in each candidate set
                    valid_candidates_list = []
                    valid_candidates_list = valid_candidates_list + valid_pair_list_1
                    valid_candidates_list = valid_candidates_list + valid_pair_list_2
                    valid_candidates_list = valid_candidates_list + valid_pair_list_3
                    valid_candidates_list = valid_candidates_list + valid_pair_list_4
                    print('here')
                    ranked_candidates = self.rank_candidates_by_fluency_score(seeker_tokens, valid_candidates_list)
                    for index, candi, in enumerate(ranked_candidates):
                        generated_dialogs.append(candi[0]+ '|score:'+ str(candi[1]))

        return generated_dialogs

    """
    this function train a bi-gram model on ground_truth responses
    to calculate fluency score for candidates 
    """
    def rank_candidates_by_fluency_score(self, seeker_tokens, unique_response_list):
        ranked_response_list = []
        for index, candid, in enumerate(unique_response_list):
            resp_score_list_temp = []
            response = candid.split('~')[1].strip().lower()
            response = self.preprocess_sentences(response)
            candidate_token_list = word_tokenize(response)
            bigrams = list(ngrams(candidate_token_list, 2))
            if self.MLE == None:
                self.MLE = Cacululate_MLE_Probs(2, corpus_file=None, cache=True)
            probability = self.MLE.sentence_probability(response, n=2)
            avg_score = probability/len(bigrams)
            print(avg_score)
            movie_context = ['movie','movies', 'movieid']
            chit_chat_tokens= ['thanks', 'bye', 'goodbye', 'thank']
            common_tokens_sk = list(set(seeker_tokens).intersection(movie_context))
            common_tokens_gt = list(set(candidate_token_list).intersection(movie_context))
            common_pref_tokens_gt = list(set(candidate_token_list).intersection(self.pref_keywords))
            common_pref_tokens_sk = list(set(seeker_tokens).intersection(self.pref_keywords))
            common_chit_chat_tokens = list(set(chit_chat_tokens).intersection(seeker_tokens))

            # if there is no 'movie' type tokens in both seeker and candidate, increase the score of candidate
            if len(common_tokens_sk) == 0 and len(common_tokens_gt) == 0 and len(common_chit_chat_tokens) == 0:
                avg_score = avg_score + 1.0

            #if there is movie tokens in both seeker and candidate utterances but is not a chit-chit response
            if len(common_tokens_sk) > 0 and len(common_tokens_gt) > 0 and len(common_chit_chat_tokens) == 0:
                avg_score = avg_score + 1.0

            #if there is common preference token between seeker and candidate, rank it at first position
            if len(common_pref_tokens_gt) > 0 and len(common_pref_tokens_sk) > 0 and len(common_chit_chat_tokens) == 0:
                avg_score = avg_score + 5.0
            if len(common_chit_chat_tokens) > 0:
                avg_score =avg_score + 2.0
            resp_score_list_temp.append(candid)
            resp_score_list_temp.append(avg_score)
            ranked_response_list.append(resp_score_list_temp)
        ranked_response_list.sort(key= lambda x: x[1], reverse=True)
        return ranked_response_list

    #funtion to discard outliers computed using mutual similarity score using BERT model
    def filter_outliers_from_set(self, candidates_set, nb_candidates):
        # this function creates pairwise combinations of candidates,
        # compute similarity score with bert embeddings,
        # and return only valid nb of candidate pairs,
        # e.g., in case of 5 candidates, there will be 10 unique pairs, only two will be retained
        valid_candidate_pairs = []
        if len(candidates_set) != 0:
            pair_order_list = list(itertools.combinations(candidates_set,2)) # create pairwise combinations from retrieved candidates
            valid_nb_pairs = math.floor(len(pair_order_list)/nb_candidates) # compute how many pairs need to be retained in the final stage
            score_pair_list = list(map(list, pair_order_list)) # convert list of tuples to list of lists
            for index, pair, in enumerate(score_pair_list):
                GT1 = pair[0].split('~')[1].strip()
                gt1_processed = self.preprocess_sentences(GT1)
                GT2 = pair[1].split('~')[1].strip()
                gt2_processed = self.preprocess_sentences(GT2)
                vectorizer = Vectorizer()
                vectorizer.bert([gt1_processed,gt2_processed])
                vectors_bert = vectorizer.vectors
                dist_1 = spatial.distance.cosine(vectors_bert[0], vectors_bert[1]) # compute spatial distance between two sentences using bert model
                score_pair_list[index].append(round(dist_1,4))
                print('dist_1: {0}'.format(dist_1))

            ## sort list based on scores after all the pairswise scores are calculated
            score_pair_list.sort(key= lambda x: x[2], reverse= True)
            valid_candidate_pairs = score_pair_list[:valid_nb_pairs] # retaine only valid nb of pairs with scores
            only_response_list= []
            for index, candidate, in enumerate(valid_candidate_pairs):
                only_response_list.append(candidate[0]) # candidate
                only_response_list.append(candidate[1]) #score
            unique_response_list = list(set(only_response_list))
        return unique_response_list



if __name__ == '__main__':
    obj = RetrieveCandidateResponses_TFIDF()
    dialogs =obj.process_dialogs(5) # parameter: nb of candidates to retrieve from retrieval model
    with open(obj.DATA_path+'Retrieved_dialogs_test.txt', 'w', encoding='utf-8') as filehandle:
        for line in dialogs:
            filehandle.writelines("%s\n" % line)
    print('execution finshed')
    exit()

