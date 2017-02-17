import matplotlib
matplotlib.use('Agg')
import os
import sys
import numpy as np
import json
import matplotlib.pyplot as plt
from write_to_log import write_log

import caffe
from caffe import layers as L
from caffe import params as P

from vqa_data_provider_layer import VQADataProvider
from visualize_tools import exec_validation, drawgraph
import config


def qlstm(mode, batchsize, T, T_c, question_c_vocab_size, question_vocab_size):
    n = caffe.NetSpec()
    mode_str = json.dumps({'mode':mode, 'batchsize':batchsize})
    n.data, n.cont, n.data1, n.cont1, n.img_feature, n.label = L.Python(\
        module='vqa_data_provider_layer', layer='VQADataProviderLayer', param_str=mode_str, ntop=6)#5 )
    
    # char embedding
    n.embed_c = L.Embed(n.data1, input_dim=question_c_vocab_size, num_output=100, \
         weight_filler=dict(type='uniform',min=-0.08,max=0.08))
    n.embed_c_scale = L.Scale(n.embed_c, n.cont1, scale_param=dict(dict(axis=0)))
    n.embed_c_scale_resh = L.Reshape(n.embed_c_scale,reshape_param=dict(shape=dict(dim=[batchsize,T_c*T,-1,1]))) # N x T_c*T x d_c x 1
    tops = L.Slice(n.embed_c_scale_resh, ntop=T, slice_param={'axis':1}) # N x T_c x d_c x 1
    for i in xrange(T):
        n.__setattr__('Slice_'+str(i+1), tops[int(i)])
    
    # avg of char embedding
    n.c_embed_1 = L.Convolution(n.Slice_1, convolution_param={'kernel_size': 1, 'num_output': 1, 'bias_term': False, 'weight_filler': dict(type='constant', value=1)}, param=dict(lr_mult=0, decay_mult=0)) # N x 1 x d_c x 1
    n.c_embed_2 = L.Convolution(n.Slice_2, convolution_param={'kernel_size': 1, 'num_output': 1, 'bias_term': False, 'weight_filler': dict(type='constant', value=1)}, param=dict(lr_mult=0, decay_mult=0)) # N x 1 x d_c x 1
    n.c_embed_3 = L.Convolution(n.Slice_3, convolution_param={'kernel_size': 1, 'num_output': 1, 'bias_term': False, 'weight_filler': dict(type='constant', value=1)}, param=dict(lr_mult=0, decay_mult=0)) # N x 1 x d_c x 1
    n.c_embed_4 = L.Convolution(n.Slice_4, convolution_param={'kernel_size': 1, 'num_output': 1, 'bias_term': False, 'weight_filler': dict(type='constant', value=1)}, param=dict(lr_mult=0, decay_mult=0)) # N x 1 x d_c x 1
    n.c_embed_5 = L.Convolution(n.Slice_5, convolution_param={'kernel_size': 1, 'num_output': 1, 'bias_term': False, 'weight_filler': dict(type='constant', value=1)}, param=dict(lr_mult=0, decay_mult=0)) # N x 1 x d_c x 1
    n.c_embed_6 = L.Convolution(n.Slice_6, convolution_param={'kernel_size': 1, 'num_output': 1, 'bias_term': False, 'weight_filler': dict(type='constant', value=1)}, param=dict(lr_mult=0, decay_mult=0)) # N x 1 x d_c x 1
    n.c_embed_7 = L.Convolution(n.Slice_7, convolution_param={'kernel_size': 1, 'num_output': 1, 'bias_term': False, 'weight_filler': dict(type='constant', value=1)}, param=dict(lr_mult=0, decay_mult=0)) # N x 1 x d_c x 1
    n.c_embed_8 = L.Convolution(n.Slice_8, convolution_param={'kernel_size': 1, 'num_output': 1, 'bias_term': False, 'weight_filler': dict(type='constant', value=1)}, param=dict(lr_mult=0, decay_mult=0)) # N x 1 x d_c x 1
    n.c_embed_9 = L.Convolution(n.Slice_9, convolution_param={'kernel_size': 1, 'num_output': 1, 'bias_term': False, 'weight_filler': dict(type='constant', value=1)}, param=dict(lr_mult=0, decay_mult=0)) # N x 1 x d_c x 1
    n.c_embed_10 = L.Convolution(n.Slice_10, convolution_param={'kernel_size': 1, 'num_output': 1, 'bias_term': False, 'weight_filler': dict(type='constant', value=1)}, param=dict(lr_mult=0, decay_mult=0)) # N x 1 x d_c x 1
    n.c_embed_11 = L.Convolution(n.Slice_11, convolution_param={'kernel_size': 1, 'num_output': 1, 'bias_term': False, 'weight_filler': dict(type='constant', value=1)}, param=dict(lr_mult=0, decay_mult=0)) # N x 1 x d_c x 1
    n.c_embed_12 = L.Convolution(n.Slice_12, convolution_param={'kernel_size': 1, 'num_output': 1, 'bias_term': False, 'weight_filler': dict(type='constant', value=1)}, param=dict(lr_mult=0, decay_mult=0)) # N x 1 x d_c x 1
    n.c_embed_13 = L.Convolution(n.Slice_13, convolution_param={'kernel_size': 1, 'num_output': 1, 'bias_term': False, 'weight_filler': dict(type='constant', value=1)}, param=dict(lr_mult=0, decay_mult=0)) # N x 1 x d_c x 1
    n.c_embed_14 = L.Convolution(n.Slice_14, convolution_param={'kernel_size': 1, 'num_output': 1, 'bias_term': False, 'weight_filler': dict(type='constant', value=1)}, param=dict(lr_mult=0, decay_mult=0)) # N x 1 x d_c x 1
    n.c_embed_15 = L.Convolution(n.Slice_15, convolution_param={'kernel_size': 1, 'num_output': 1, 'bias_term': False, 'weight_filler': dict(type='constant', value=1)}, param=dict(lr_mult=0, decay_mult=0)) # N x 1 x d_c x 1
    n.c_embed_16 = L.Convolution(n.Slice_16, convolution_param={'kernel_size': 1, 'num_output': 1, 'bias_term': False, 'weight_filler': dict(type='constant', value=1)}, param=dict(lr_mult=0, decay_mult=0)) # N x 1 x d_c x 1
    n.c_embed_17 = L.Convolution(n.Slice_17, convolution_param={'kernel_size': 1, 'num_output': 1, 'bias_term': False, 'weight_filler': dict(type='constant', value=1)}, param=dict(lr_mult=0, decay_mult=0)) # N x 1 x d_c x 1
    n.c_embed_18 = L.Convolution(n.Slice_18, convolution_param={'kernel_size': 1, 'num_output': 1, 'bias_term': False, 'weight_filler': dict(type='constant', value=1)}, param=dict(lr_mult=0, decay_mult=0)) # N x 1 x d_c x 1
    n.c_embed_19 = L.Convolution(n.Slice_19, convolution_param={'kernel_size': 1, 'num_output': 1, 'bias_term': False, 'weight_filler': dict(type='constant', value=1)}, param=dict(lr_mult=0, decay_mult=0)) # N x 1 x d_c x 1
    n.c_embed_20 = L.Convolution(n.Slice_20, convolution_param={'kernel_size': 1, 'num_output': 1, 'bias_term': False, 'weight_filler': dict(type='constant', value=1)}, param=dict(lr_mult=0, decay_mult=0)) # N x 1 x d_c x 1
    n.c_embed_21 = L.Convolution(n.Slice_21, convolution_param={'kernel_size': 1, 'num_output': 1, 'bias_term': False, 'weight_filler': dict(type='constant', value=1)}, param=dict(lr_mult=0, decay_mult=0)) # N x 1 x d_c x 1
    n.c_embed_22 = L.Convolution(n.Slice_22, convolution_param={'kernel_size': 1, 'num_output': 1, 'bias_term': False, 'weight_filler': dict(type='constant', value=1)}, param=dict(lr_mult=0, decay_mult=0)) # N x 1 x d_c x 1

    concat_c_embed = [n.c_embed_1, n.c_embed_2, n.c_embed_3, n.c_embed_4, n.c_embed_5, n.c_embed_6, n.c_embed_7, n.c_embed_8, n.c_embed_9, n.c_embed_10,\
     n.c_embed_11, n.c_embed_12, n.c_embed_13, n.c_embed_14, n.c_embed_15, n.c_embed_16, n.c_embed_17, n.c_embed_18, n.c_embed_19, n.c_embed_20, n.c_embed_21, n.c_embed_22]
    n.concat_char_embed = L.Concat(*concat_c_embed, concat_param={'axis': 1}) # N x T x d_c x 1
    n.concat_char_embed_resh = L.Reshape(n.concat_char_embed,reshape_param=dict(shape=dict(dim=[batchsize,T,-1])))
    
    # word embedding
    n.embed_w = L.Embed(n.data, input_dim=question_vocab_size, num_output=200, \
        weight_filler=dict(type='uniform',min=-0.08,max=0.08)) # N x T x d_w

    # combine word and char embedding
    concat_word_embed = [n.embed_w, n.concat_char_embed_resh]
    n.concat_embed = L.Concat(*concat_word_embed, concat_param={'axis': 2}) # N x T x (d_c+d_w)

    n.embed_scale = L.Scale(n.concat_embed, n.cont, scale_param=dict(dict(axis=0))) # N x T x (d_c+d_w)
    n.embed_scale_resh = L.Reshape(n.embed_scale,reshape_param=dict(shape=dict(dim=[batchsize,T,-1,1]))) # N x T x (d_c+d_w) x 1

    # avg of word embedding
    n.embed_avg = L.Convolution(n.embed_scale_resh, convolution_param={'kernel_size': 1, 'num_output': 1, 'bias_term': False, 'weight_filler': dict(type='constant', value=1)}, param=dict(lr_mult=0, decay_mult=0)) # N x 1 x (d_c+d_w) x 1
    n.embed_avg_resh = L.Reshape(n.embed_avg,reshape_param=dict(shape=dict(dim=[batchsize,300,1,1])))

    n.q_emb_tanh_droped_resh_tiled_1 = L.Tile(n.embed_avg_resh, axis=2, tiles=14)
    n.q_emb_tanh_droped_resh_tiled = L.Tile(n.q_emb_tanh_droped_resh_tiled_1, axis=3, tiles=14)
    n.i_emb_tanh_droped_resh = L.Reshape(n.img_feature,reshape_param=dict(shape=dict(dim=[-1,2048,14,14])))
    n.blcf = L.CompactBilinear(n.q_emb_tanh_droped_resh_tiled, n.i_emb_tanh_droped_resh, compact_bilinear_param=dict(num_output=16000,sum_pool=False))
    n.blcf_sign_sqrt = L.SignedSqrt(n.blcf)
    n.blcf_sign_sqrt_l2 = L.L2Normalize(n.blcf_sign_sqrt)
    n.blcf_droped = L.Dropout(n.blcf_sign_sqrt_l2,dropout_param={'dropout_ratio':0.1})

    # multi-channel attention
    n.att_conv1 = L.Convolution(n.blcf_droped, kernel_size=1, stride=1, num_output=512, pad=0, weight_filler=dict(type='xavier'))
    n.att_conv1_relu = L.ReLU(n.att_conv1)
    n.att_conv2 = L.Convolution(n.att_conv1_relu, kernel_size=1, stride=1, num_output=2, pad=0, weight_filler=dict(type='xavier'))
    n.att_reshaped = L.Reshape(n.att_conv2,reshape_param=dict(shape=dict(dim=[-1,2,14*14])))
    n.att_softmax = L.Softmax(n.att_reshaped, axis=2)
    n.att = L.Reshape(n.att_softmax,reshape_param=dict(shape=dict(dim=[-1,2,14,14])))
    att_maps = L.Slice(n.att, ntop=2, slice_param={'axis':1})
    n.att_map0 = att_maps[0]
    n.att_map1 = att_maps[1]
    dummy = L.DummyData(shape=dict(dim=[batchsize, 1]), data_filler=dict(type='constant', value=1), ntop=1)
    n.att_feature0  = L.SoftAttention(n.i_emb_tanh_droped_resh, n.att_map0, dummy)
    n.att_feature1  = L.SoftAttention(n.i_emb_tanh_droped_resh, n.att_map1, dummy)
    n.att_feature0_resh = L.Reshape(n.att_feature0, reshape_param=dict(shape=dict(dim=[-1,2048])))
    n.att_feature1_resh = L.Reshape(n.att_feature1, reshape_param=dict(shape=dict(dim=[-1,2048])))
    n.att_feature = L.Concat(n.att_feature0_resh, n.att_feature1_resh)

    # merge attention and lstm with compact bilinear pooling
    n.att_feature_resh = L.Reshape(n.att_feature, reshape_param=dict(shape=dict(dim=[-1,4096,1,1])))
    #n.lstm_12_resh = L.Reshape(n.lstm_12, reshape_param=dict(shape=dict(dim=[-1,2048,1,1])))
    n.bc_att_lstm = L.CompactBilinear(n.att_feature_resh, n.embed_avg_resh, 
                                      compact_bilinear_param=dict(num_output=16000,sum_pool=False))
    n.bc_sign_sqrt = L.SignedSqrt(n.bc_att_lstm)
    n.bc_sign_sqrt_l2 = L.L2Normalize(n.bc_sign_sqrt)

    n.bc_dropped = L.Dropout(n.bc_sign_sqrt_l2, dropout_param={'dropout_ratio':0.1})
    n.bc_dropped_resh = L.Reshape(n.bc_dropped, reshape_param=dict(shape=dict(dim=[-1, 16000])))

    n.prediction = L.InnerProduct(n.bc_dropped_resh, num_output=3000, weight_filler=dict(type='xavier'))
    n.loss = L.SoftmaxWithLoss(n.prediction, n.label)
    return n.to_proto()

def make_answer_vocab(adic, vocab_size):
    """
    Returns a dictionary that maps words to indices.
    """
    adict = {'':0}
    nadict = {'':1000000}
    vid = 1
    for qid in adic.keys():
        answer_obj = adic[qid]
        answer_list = [ans['answer'] for ans in answer_obj]
        
        for q_ans in answer_list:
            # create dict
            if adict.has_key(q_ans):
                nadict[q_ans] += 1
            else:
                nadict[q_ans] = 1
                adict[q_ans] = vid
                vid +=1

    # debug
    nalist = []
    for k,v in sorted(nadict.items(), key=lambda x:x[1]):
        nalist.append((k,v))

    # keep the top 3000 answers
    n_del_ans = 0
    n_valid_ans = 0
    adict_nid = {}
    for i, w in enumerate(nalist[:-vocab_size]):
        del adict[w[0]]
        n_del_ans += w[1]
    for i, w in enumerate(nalist[-vocab_size:]):
        n_valid_ans += w[1]
        adict_nid[w[0]] = i
    
    return adict_nid

def make_question_vocab(qdic):
    """
    Returns a dictionary that maps words to indices.
    """
    vdict = {'':0}
    vid = 1
    cdict = {'':0}
    cvid = 1
    for qid in qdic.keys():
        # sequence to list
        q_str = qdic[qid]['qstr']
        q_list = VQADataProvider.seq_to_list(q_str)

        # create dict
        for w in q_list:
            if not vdict.has_key(w):
                vdict[w] = vid
                vid +=1
            for c in list(w):
                if not cdict.has_key(c):
                    cdict[c] = cvid
                    cvid += 1

    return vdict, cdict

def make_vocab_files():
    """
    Produce the question and answer vocabulary files.
    """
    write_log('making question vocab... ' + config.QUESTION_VOCAB_SPACE, 'log.txt')
    write_log('making question character vocab... ' + config.QUESTION_VOCAB_SPACE, 'log.txt')
    qdic, _ = VQADataProvider.load_data(config.QUESTION_VOCAB_SPACE)
    question_vocab, question_char_vocab = make_question_vocab(qdic)
    write_log('making answer vocab... ' + config.ANSWER_VOCAB_SPACE, 'log.txt')
    _, adic = VQADataProvider.load_data(config.ANSWER_VOCAB_SPACE)
    answer_vocab = make_answer_vocab(adic, config.NUM_OUTPUT_UNITS)
    return question_vocab, question_char_vocab, answer_vocab

def main():
    if not os.path.exists('./result'):
        os.makedirs('./result')

    question_vocab, answer_vocab = {}, {}
    if os.path.exists('./result/cdict.json') and os.path.exists('./result/adict.json') and os.path.exists('./result/vdict.json'):
        write_log('restoring vocab', 'log.txt')
        with open('./result/cdict.json','r') as f:
            question_char_vocab = json.load(f)
        with open('./result/vdict.json','r') as f:
            question_vocab = json.load(f)
        with open('./result/adict.json','r') as f:
            answer_vocab = json.load(f)
    else:
        question_vocab, question_char_vocab, answer_vocab = make_vocab_files()
        with open('./result/cdict.json','w') as f:
            json.dump(question_char_vocab, f)
        with open('./result/vdict.json','w') as f:
            json.dump(question_vocab, f)
        with open('./result/adict.json','w') as f:
            json.dump(answer_vocab, f)

    write_log('question character vocab size: '+ str(len(question_char_vocab)), 'log.txt')
    write_log('question vocab size: '+ str(len(question_vocab)), 'log.txt')
    write_log('answer vocab size: '+ str(len(answer_vocab)), 'log.txt')


    with open('./result/proto_train.prototxt', 'w') as f:
        f.write(str(qlstm(config.TRAIN_DATA_SPLITS, config.BATCH_SIZE, \
            config.MAX_WORDS_IN_QUESTION, config.LENGTH_OF_LONGEST_WORD, len(question_char_vocab), len(question_vocab))))

    with open('./result/proto_test.prototxt', 'w') as f:
        f.write(str(qlstm('val', config.VAL_BATCH_SIZE, \
            config.MAX_WORDS_IN_QUESTION, config.LENGTH_OF_LONGEST_WORD, len(question_char_vocab), len(question_vocab))))

    caffe.set_device(config.GPU_ID)
    caffe.set_mode_gpu()
    solver = caffe.get_solver('./qlstm_solver.prototxt')

    train_loss = np.zeros(config.MAX_ITERATIONS)
    # results = []

    for it in range(config.MAX_ITERATIONS):
        solver.step(1)
    
        # store the train loss
        train_loss[it] = solver.net.blobs['loss'].data
   
        if it != 0 and it % config.PRINT_INTERVAL == 0:
            write_log('------------------------------------', 'log.txt')
            write_log('Iteration: ' + str(it), 'log.txt')
            c_mean_loss = train_loss[it-config.PRINT_INTERVAL:it].mean()
            write_log('Train loss: ' + str(c_mean_loss), 'log.txt')
        if it != 0 and it % config.VALIDATE_INTERVAL == 0: # acutually test
            solver.test_nets[0].save('./result/tmp.caffemodel')
            write_log('Validating...', 'log.txt')
            test_loss, acc_overall, acc_per_ques, acc_per_ans = exec_validation(config.GPU_ID, 'val', it=it)
            write_log('Iteration: ' + str(it), 'log.txt')
            write_log('Test loss: ' + str(test_loss), 'log.txt')
            write_log('Overall Accuracy: ' + str(acc_overall), 'log.txt')
            write_log('Per Question Type Accuracy is the following:', 'log.txt')
            for quesType in acc_per_ques:
                write_log("%s : %.02f" % (quesType, acc_per_ques[quesType]), 'log.txt')
            write_log('Per Answer Type Accuracy is the following:', 'log.txt')
            for ansType in acc_per_ans:
                write_log("%s : %.02f" % (ansType, acc_per_ans[ansType]), 'log.txt')
            # results.append([it, c_mean_loss, test_loss, acc_overall, acc_per_ques, acc_per_ans])
            # best_result_idx = np.array([x[3] for x in results]).argmax()
            # write_log('Best accuracy of ' + str(results[best_result_idx][3]) + ' was at iteration ' + str(results[best_result_idx][0]), 'log.txt')
            # drawgraph(results)

if __name__ == '__main__':
    main()
