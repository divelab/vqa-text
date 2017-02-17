import caffe
import numpy as np
import re, json, random
import config
import spacy
from write_to_log import write_log

QID_KEY_SEPARATOR = '/'
#GLOVE_EMBEDDING_SIZE = 300

class VQADataProvider:

    def __init__(self, batchsize=64, max_length=config.MAX_WORDS_IN_QUESTION, max_w_length=config.LENGTH_OF_LONGEST_WORD, mode='train'):
        self.batchsize = batchsize
        self.d_vocabulary = None
        self.batch_index = None
        self.batch_len = None
        self.rev_adict = None
        self.max_length = max_length
        self.max_w_length = max_w_length
        self.mode = mode
        self.qdic, self.adic = VQADataProvider.load_data(mode)

        with open('./result/cdict.json','r') as f:
            self.cdict = json.load(f)
        with open('./result/vdict.json','r') as f:
            self.vdict = json.load(f)
        with open('./result/adict.json','r') as f:
            self.adict = json.load(f)

        self.n_ans_vocabulary = len(self.adict)
        #self.nlp = spacy.load('en', vectors='en_glove_cc_300_1m_vectors')
        #self.glove_dict = {} # word -> glove vector

    @staticmethod
    def load_vqa_json(data_split):
        """
        Parses the question and answer json files for the given data split. 
        Returns the question dictionary and the answer dictionary.
        """
        qdic, adic = {}, {}

        with open(config.DATA_PATHS[data_split]['ques_file'], 'r') as f:
            qdata = json.load(f)['questions']
            for q in qdata:
                qdic[data_split + QID_KEY_SEPARATOR + str(q['question_id'])] = \
                    {'qstr': q['question'], 'iid': q['image_id']}

        if 'test' not in data_split:
            with open(config.DATA_PATHS[data_split]['ans_file'], 'r') as f:
                adata = json.load(f)['annotations']
                for a in adata:
                    adic[data_split + QID_KEY_SEPARATOR + str(a['question_id'])] = \
                        a['answers']

        write_log('parsed ' + str(len(qdic)) + ' questions for ' + data_split, 'log.txt')
        return qdic, adic

    @staticmethod
    def load_genome_json():
        """
        Parses the genome json file. Returns the question dictionary and the
        answer dictionary.
        """
        qdic, adic = {}, {}

        with open(config.DATA_PATHS['genome']['genome_file'], 'r') as f:
            qdata = json.load(f)
            for q in qdata:
                key = 'genome' + QID_KEY_SEPARATOR + str(q['id'])
                qdic[key] = {'qstr': q['question'], 'iid': q['image']}
                adic[key] = [{'answer': q['answer']}]

        write_log('parsed ' + str(len(qdic)) + ' questions for genome', 'log.txt')
        return qdic, adic

    @staticmethod
    def load_data(data_split_str):
        all_qdic, all_adic = {}, {}
        for data_split in data_split_str.split('+'):
            assert data_split in config.DATA_PATHS.keys(), 'unknown data split'
            if data_split == 'genome':
                qdic, adic = VQADataProvider.load_genome_json()
                all_qdic.update(qdic)
                all_adic.update(adic)
            else:
                qdic, adic = VQADataProvider.load_vqa_json(data_split)
                all_qdic.update(qdic)
                all_adic.update(adic)
        
        return all_qdic, all_adic

    def getQuesIds(self):
        return self.qdic.keys()

    def getStrippedQuesId(self, qid):
        return qid.split(QID_KEY_SEPARATOR)[1]

    def getImgId(self,qid):
        return self.qdic[qid]['iid']

    def getQuesStr(self,qid):
        return self.qdic[qid]['qstr']

    def getAnsObj(self,qid):
        if self.mode == 'test-dev' or self.mode == 'test':
            return -1
        return self.adic[qid]

    @staticmethod
    def seq_to_list(s):
        t_str = s.lower()
        for i in [r'\?',r'\!',r'\'',r'\"',r'\$',r'\:',r'\@',r'\(',r'\)',r'\,',r'\.',r'\;']:
            t_str = re.sub( i, '', t_str)
        for i in [r'\-',r'\/']:
            t_str = re.sub( i, ' ', t_str)
        q_list = re.sub(r'\?','',t_str.lower()).split(' ')
        q_list = filter(lambda x: len(x) > 0, q_list)
        return q_list

    def extract_answer(self,answer_obj):
        """ Return the most popular answer in string."""
        if self.mode == 'test-dev' or self.mode == 'test':
            return -1
        answer_list = [ answer_obj[i]['answer'] for i in xrange(10)]
        dic = {}
        for ans in answer_list:
            if dic.has_key(ans):
                dic[ans] +=1
            else:
                dic[ans] = 1
        max_key = max((v,k) for (k,v) in dic.items())[1]
        return max_key

    def extract_answer_prob(self,answer_obj):
        """ Return the most popular answer in string."""
        if self.mode == 'test-dev' or self.mode == 'test':
            return -1

        answer_list = [ ans['answer'] for ans in answer_obj]
        prob_answer_list = []
        for ans in answer_list:
            if self.adict.has_key(ans):
                prob_answer_list.append(ans)

        if len(prob_answer_list) == 0:
            if self.mode == 'val' or self.mode == 'test-dev' or self.mode == 'test':
                return 'hoge'
            else:
                raise Exception("This should not happen.")
        else:
            return random.choice(prob_answer_list)
 
    # def qlist_to_vec(self, max_length, q_list):
    #     """
    #     Converts a list of words into a format suitable for the embedding layer.

    #     Arguments:
    #     max_length -- the maximum length of a question sequence
    #     q_list -- a list of words which are the tokens in the question

    #     Returns:
    #     qvec -- A max_length length vector containing one-hot indices for each word
    #     cvec -- A max_length length sequence continuation indicator vector
    #     glove_matrix -- A max_length x GLOVE_EMBEDDING_SIZE matrix containing the glove embedding for
    #         each word
    #     """
    #     qvec = np.zeros(max_length)
    #     cvec = np.zeros(max_length)
    #     glove_matrix = np.zeros(max_length * GLOVE_EMBEDDING_SIZE).reshape(max_length, GLOVE_EMBEDDING_SIZE)
    #     for i in xrange(max_length):
    #         if i < max_length - len(q_list):
    #             cvec[i] = 0
    #         else:
    #             w = q_list[i-(max_length-len(q_list))]
    #             if w not in self.glove_dict:
    #                 self.glove_dict[w] = self.nlp(u'%s' % w).vector
    #             glove_matrix[i] = self.glove_dict[w]
    #             # is the word in the vocabulary?
    #             if self.vdict.has_key(w) is False:
    #                 w = ''
    #             qvec[i] = self.vdict[w]
    #             cvec[i] = 1 #0 if i == max_length - len(q_list) else 1
    #             # UPDATE_10/12:
    #             #It seems that the original way is right for lstm.
    #             '''
    #             This is from https://github.com/BVLC/caffe/pull/1873.
    #             "delta_{t,n} should be a binary indicator (i.e., value in {0, 1}),
    #             where a value of 0 means that timestep t of stream n is the beginning of a new sequence,
    #             and a value of 1 means that timestep t of stream n is continuing the sequence from timestep t-1 of stream n.
    #             Under the hood, the previous timestep's hidden state is multiplied by these delta values.
    #             The fact that these indicators are specified on a per-timestep and per-stream basis allows for streams of arbitrary different lengths without any padding or truncation.
    #             At the beginning of the forward pass, the final hidden state from the previous forward pass (h_T) is copied into the initial hidden state for the new forward pass (h_0),
    #             allowing for exact inference across arbitrarily long sequences, even if T == 1.
    #             However, if any sequences cross batch boundaries, backpropagation through time is approximate -- it is truncated along the batch boundaries."
    #             '''
    #             # I think the original way is false for CNN here. I want to keep the word embedding for unknown word as zero vector.
    #             # For example, if we have a sentence of length 7, there would be only 6 1's in cvec.

    #     return qvec, cvec, glove_matrix
        
    def qlist_to_vec(self, max_length, max_w_length, q_list):
        """
        Converts a list of chars into a format suitable for the embedding layer.

        Arguments:
        max_length -- the maximum length of a question sequence
        max_w_length -- the maximum length of a character sequence for each word
        q_list -- a list of words which are the tokens in the question

        Returns:
        qvec -- A max_length length vector containing one-hot indices for each word
        cvec -- A max_length length sequence continuation indicator vector
        qcvec -- A max_w_length*max_length length vector containing one-hot indices for each char
        ccvec -- A max_w_length*max_length length sequence continuation indicator vector
        """
        qvec = np.zeros(max_length)
        cvec = np.zeros(max_length)
        qcvec = np.zeros(max_w_length*max_length)
        ccvec = np.zeros(max_w_length*max_length)
        for i in xrange(max_length):
            if i < max_length - len(q_list):
                cvec[i] = 0
            else:
                w = q_list[i-(max_length-len(q_list))]
                # is the word in the vocabulary?
                if self.vdict.has_key(w) is False:
                    w = ''
                qvec[i] = self.vdict[w]
                cvec[i] = 1.0 / len(q_list)
                j = max_w_length * i
                w_list = list(w)
                for k in xrange(max_w_length):
                    if k < max_w_length - len(w_list):
                        ccvec[j+k] = 0
                    else:
                        c = w_list[k-(max_w_length-len(w_list))]
                        # is the char in the vocabulary?
                        if self.cdict.has_key(c) is False:
                            c = ''
                        qcvec[j+k] = self.cdict[c]
                        ccvec[j+k] = 1.0 / len(w_list)
        return qvec, cvec, qcvec, ccvec

    def answer_to_vec(self, ans_str):
        """ Return answer id if the answer is included in vocabulary otherwise '' """
        if self.mode =='test-dev' or self.mode == 'test':
            return -1

        if self.adict.has_key(ans_str):
            ans = self.adict[ans_str]
        else:
            ans = self.adict['']
        return ans
 
    def vec_to_answer(self, ans_symbol):
        """ Return answer id if the answer is included in vocabulary otherwise '' """
        if self.rev_adict is None:
            rev_adict = {}
            for k,v in self.adict.items():
                rev_adict[v] = k
            self.rev_adict = rev_adict

        return self.rev_adict[ans_symbol]
 
    def create_batch(self,qid_list):

        qvec = (np.zeros(self.batchsize*self.max_length)).reshape(self.batchsize,self.max_length)
        cvec = (np.zeros(self.batchsize*self.max_length)).reshape(self.batchsize,self.max_length)
        qcvec = (np.zeros(self.batchsize*self.max_w_length*self.max_length)).reshape(self.batchsize,self.max_w_length*self.max_length)
        ccvec = (np.zeros(self.batchsize*self.max_w_length*self.max_length)).reshape(self.batchsize,self.max_w_length*self.max_length)
        ivec = (np.zeros(self.batchsize*2048*14*14)).reshape(self.batchsize,2048,14,14)
        avec = (np.zeros(self.batchsize)).reshape(self.batchsize)
        #glove_matrix = np.zeros(self.batchsize * self.max_length * GLOVE_EMBEDDING_SIZE).reshape(\
            #self.batchsize, self.max_length, GLOVE_EMBEDDING_SIZE)

        for i,qid in enumerate(qid_list):

            # load raw question information
            q_str = self.getQuesStr(qid)
            q_ans = self.getAnsObj(qid)
            q_iid = self.getImgId(qid)

            # convert question to vec
            q_list = VQADataProvider.seq_to_list(q_str)
            t_qvec, t_cvec, t_qcvec, t_ccvec = self.qlist_to_vec(self.max_length, self.max_w_length, q_list)

            try:
                qid_split = qid.split(QID_KEY_SEPARATOR)
                data_split = qid_split[0]
                if data_split == 'genome':
                    t_ivec = np.load(config.DATA_PATHS['genome']['features_prefix'] + str(q_iid) + '.jpg.npz')['x']
                else:
                    t_ivec = np.load(config.DATA_PATHS[data_split]['features_prefix'] + str(q_iid).zfill(12) + '.jpg.npz')['x']
                t_ivec = ( t_ivec / np.sqrt((t_ivec**2).sum()) )
            except:
                t_ivec = 0.
                write_log('data not found for qid : ' + str(q_iid) + ' ' + self.mode, 'log.txt')
             
            # convert answer to vec
            if self.mode == 'val' or self.mode == 'test-dev' or self.mode == 'test':
                q_ans_str = self.extract_answer(q_ans)
            else:
                q_ans_str = self.extract_answer_prob(q_ans)
            t_avec = self.answer_to_vec(q_ans_str)

            qvec[i,...] = t_qvec
            cvec[i,...] = t_cvec
            qcvec[i,...] = t_qcvec
            ccvec[i,...] = t_ccvec
            ivec[i,...] = t_ivec
            avec[i,...] = t_avec
            #glove_matrix[i,...] = t_glove_matrix

        return qvec, cvec, qcvec, ccvec, ivec, avec#, glove_matrix

 
    def get_batch_vec(self):
        if self.batch_len is None:
            self.n_skipped = 0
            qid_list = self.getQuesIds()
            random.shuffle(qid_list)
            self.qid_list = qid_list
            self.batch_len = len(qid_list)
            self.batch_index = 0
            self.epoch_counter = 0

        def has_at_least_one_valid_answer(t_qid):
            answer_obj = self.getAnsObj(t_qid)
            answer_list = [ans['answer'] for ans in answer_obj]
            for ans in answer_list:
                if self.adict.has_key(ans):
                    return True

        counter = 0
        t_qid_list = []
        t_iid_list = []
        while counter < self.batchsize:
            t_qid = self.qid_list[self.batch_index]
            t_iid = self.getImgId(t_qid)
            if self.mode == 'val' or self.mode == 'test-dev' or self.mode == 'test':
                t_qid_list.append(t_qid)
                t_iid_list.append(t_iid)
                counter += 1
            elif has_at_least_one_valid_answer(t_qid):
                t_qid_list.append(t_qid)
                t_iid_list.append(t_iid)
                counter += 1
            else:
                self.n_skipped += 1 

            if self.batch_index < self.batch_len-1:
                self.batch_index += 1
            else:
                self.epoch_counter += 1
                qid_list = self.getQuesIds()
                random.shuffle(qid_list)
                self.qid_list = qid_list
                self.batch_index = 0
                write_log("%d questions were skipped in a single epoch" % self.n_skipped, 'log.txt')
                self.n_skipped = 0

        t_batch = self.create_batch(t_qid_list)
        return t_batch + (t_qid_list, t_iid_list, self.epoch_counter)

class VQADataProviderLayer(caffe.Layer):
    """
    Provide input data for VQA.
    """

    def setup(self, bottom, top):
        self.batchsize = json.loads(self.param_str)['batchsize']
        self.top_names = ['data','cont', 'data1', 'cont1', 'feature','label']#,'glove']
        top[0].reshape(self.batchsize,config.MAX_WORDS_IN_QUESTION)
        top[1].reshape(self.batchsize,config.MAX_WORDS_IN_QUESTION)
        top[2].reshape(self.batchsize,config.LENGTH_OF_LONGEST_WORD*config.MAX_WORDS_IN_QUESTION)
        top[3].reshape(self.batchsize,config.LENGTH_OF_LONGEST_WORD*config.MAX_WORDS_IN_QUESTION)
        top[4].reshape(self.batchsize,2048,14,14)
        top[5].reshape(self.batchsize)
        #top[4].reshape(self.batchsize,config.MAX_WORDS_IN_QUESTION,GLOVE_EMBEDDING_SIZE)

        self.mode = json.loads(self.param_str)['mode']
        if self.mode == 'val' or self.mode == 'test-dev' or self.mode == 'test':
            pass
        else:
            self.dp = VQADataProvider(batchsize=self.batchsize, mode=self.mode)

    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        if self.mode == 'val' or self.mode == 'test-dev' or self.mode == 'test':
            pass
        else:
            word, cont, word_c, cont_c, feature, answer, _, _, _ = self.dp.get_batch_vec()
            top[0].data[...] = word # np.transpose(word,(1,0)) # N x T -> T x N
            top[1].data[...] = cont # np.transpose(cont,(1,0))
            top[2].data[...] = word_c
            top[3].data[...] = cont_c
            top[4].data[...] = feature
            top[5].data[...] = answer
            #top[4].data[...] = glove_matrix # np.transpose(glove_matrix, (1,0,2)) # N x T x 300 -> T x N x 300

    def backward(self, top, propagate_down, bottom):
        pass

