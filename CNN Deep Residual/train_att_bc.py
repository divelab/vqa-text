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


def qlstm(mode, batchsize, T, question_vocab_size):
    n = caffe.NetSpec()
    mode_str = json.dumps({'mode':mode, 'batchsize':batchsize})
    n.data, n.cont, n.img_feature, n.label = L.Python(\
        module='vqa_data_provider_layer', layer='VQADataProviderLayer', param_str=mode_str, ntop=4)#5 )
    
    # # word embedding (static + dynamic)
    # n.embed_ba = L.Embed(n.data, input_dim=question_vocab_size, num_output=300, \
    #     weight_filler=dict(type='uniform',min=-0.08,max=0.08))
    # n.embed_scale = L.Scale(n.embed_ba, n.cont, scale_param=dict(dict(axis=0)))
    # n.embed_scale_resh = L.Reshape(n.embed_scale,\
    #                       reshape_param=dict(\
    #                           shape=dict(dim=[batchsize,1,T,300])))
    # n.glove_scale = L.Scale(n.glove, n.cont, scale_param=dict(dict(axis=0)))
    # n.glove_scale_resh = L.Reshape(n.glove_scale,\
    #                       reshape_param=dict(\
    #                           shape=dict(dim=[batchsize,1,T,300])))
    # concat_word_embed = [n.embed_scale_resh, n.glove_scale_resh]
    # n.concat_embed = L.Concat(*concat_word_embed, concat_param={'axis': 1}) # N x 2 x T x 300

    # char embedding
    n.embed_ba = L.Embed(n.data, input_dim=question_vocab_size, num_output=50, \
         weight_filler=dict(type='uniform',min=-0.08,max=0.08))
    n.embed_scale = L.Scale(n.embed_ba, n.cont, scale_param=dict(dict(axis=0)))
    n.embed_scale_resh = L.Reshape(n.embed_scale,\
                           reshape_param=dict(\
                               shape=dict(dim=[batchsize,1,T,50])))
    
    # char deep convolution
    n.char_conv_1 = L.Convolution(n.embed_scale_resh, kernel_h=5, kernel_w=50, stride=1, num_output=256, weight_filler=dict(type='gaussian', std=0.05)) # N x 1 x 100 x 50 -> N x 256 x 96 x 1
    n.char_relu_1 = L.ReLU(n.char_conv_1)
    n.char_pool_1 = L.Pooling(n.char_relu_1, kernel_h=2, kernel_w=1, stride=2, pool=P.Pooling.MAX) # N x 256 x 96 x 1 -> N x 256 x 48 x 1
    n.char_conv_2 = L.Convolution(n.char_pool_1, kernel_h=5, kernel_w=1, stride=1, num_output=256, weight_filler=dict(type='gaussian', std=0.05)) # N x 256 x 48 x 1 -> N x 256 x 44 x 1
    n.char_relu_2 = L.ReLU(n.char_conv_2)
    n.char_pool_2 = L.Pooling(n.char_relu_2, kernel_h=2, kernel_w=1, stride=2, pool=P.Pooling.MAX) # N x 256 x 44 x 1 -> N x 256 x 22 x 1
    n.char_conv_3 = L.Convolution(n.char_pool_2, kernel_h=3, kernel_w=1, stride=1, num_output=256, weight_filler=dict(type='gaussian', std=0.05)) # N x 256 x 22 x 1 -> N x 256 x 20 x 1
    n.char_relu_3 = L.ReLU(n.char_conv_3)
    n.char_conv_4 = L.Convolution(n.char_relu_3, kernel_h=3, kernel_w=1, stride=1, num_output=256, weight_filler=dict(type='gaussian', std=0.05)) # N x 256 x 20 x 1 -> N x 256 x 18 x 1
    n.char_relu_4 = L.ReLU(n.char_conv_4)
    n.char_conv_5 = L.Convolution(n.char_relu_4, kernel_h=3, kernel_w=1, stride=1, num_output=256, weight_filler=dict(type='gaussian', std=0.05)) # N x 256 x 18 x 1 -> N x 256 x 16 x 1
    n.char_relu_5 = L.ReLU(n.char_conv_5)
    n.char_pool_3 = L.Pooling(n.char_relu_5, kernel_h=2, kernel_w=1, stride=2, pool=P.Pooling.MAX) # N x 256 x 16 x 1 -> N x 256 x 8 x 1
    n.vec_reshape = L.Reshape(n.char_pool_3,reshape_param=dict(shape=dict(dim=[-1,2048,1,1])))
    n.concat_vec_dropped = L.Dropout(n.vec_reshape,dropout_param={'dropout_ratio':0.5})

    n.q_emb_tanh_droped_resh_tiled_1 = L.Tile(n.concat_vec_dropped, axis=2, tiles=14)
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
    n.bc_att_lstm = L.CompactBilinear(n.att_feature_resh, n.concat_vec_dropped, 
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

# def make_question_vocab(qdic):
#     """
#     Returns a dictionary that maps words to indices.
#     """
#     vdict = {'':0}
#     vid = 1
#     for qid in qdic.keys():
#         # sequence to list
#         q_str = qdic[qid]['qstr']
#         q_list = VQADataProvider.seq_to_list(q_str)

#         # create dict
#         for w in q_list:
#             if not vdict.has_key(w):
#                 vdict[w] = vid
#                 vid +=1

#     return vdict

def make_question_char_vocab(qdic):
    """
    Returns a dictionary that maps characters to indices.
    """
    cdict = {'':0}
    vid = 1
    for qid in qdic.keys():
        q_str = qdic[qid]['qstr']
        q_char_list = VQADataProvider.seq_to_char_list(q_str)

        for c in q_char_list:
            if not cdict.has_key(c):
                cdict[c] = vid
                vid += 1

    return cdict

def make_vocab_files():
    """
    Produce the question and answer vocabulary files.
    """
    #write_log('making question vocab... ' + config.QUESTION_VOCAB_SPACE, 'log.txt')
    write_log('making question character vocab... ' + config.QUESTION_VOCAB_SPACE, 'log.txt')
    qdic, _ = VQADataProvider.load_data(config.QUESTION_VOCAB_SPACE)
    #question_vocab = make_question_vocab(qdic)
    question_char_vocab = make_question_char_vocab(qdic)
    write_log('making answer vocab... ' + config.ANSWER_VOCAB_SPACE, 'log.txt')
    _, adic = VQADataProvider.load_data(config.ANSWER_VOCAB_SPACE)
    answer_vocab = make_answer_vocab(adic, config.NUM_OUTPUT_UNITS)
    return question_char_vocab, answer_vocab
    #return question_vocab, question_char_vocab, answer_vocab

def main():
    if not os.path.exists('./result'):
        os.makedirs('./result')

    question_vocab, answer_vocab = {}, {}
    if os.path.exists('./result/cdict.json') and os.path.exists('./result/adict.json'):# and os.path.exists('./result/vdict.json'):
        write_log('restoring vocab', 'log.txt')
        with open('./result/cdict.json','r') as f:
            question_char_vocab = json.load(f)
        #with open('./result/vdict.json','r') as f:
            #question_vocab = json.load(f)
        with open('./result/adict.json','r') as f:
            answer_vocab = json.load(f)
    else:
        question_char_vocab, answer_vocab = make_vocab_files()
        with open('./result/cdict.json','w') as f:
            json.dump(question_char_vocab, f)
        #with open('./result/vdict.json','w') as f:
            #json.dump(question_vocab, f)
        with open('./result/adict.json','w') as f:
            json.dump(answer_vocab, f)

    write_log('question character vocab size: '+ str(len(question_char_vocab)), 'log.txt')
    #write_log('question vocab size: '+ str(len(question_vocab)), 'log.txt')
    write_log('answer vocab size: '+ str(len(answer_vocab)), 'log.txt')


    with open('./result/proto_train.prototxt', 'w') as f:
        f.write(str(qlstm(config.TRAIN_DATA_SPLITS, config.BATCH_SIZE, \
            config.MAX_CHARS_IN_QUESTION, len(question_char_vocab))))

    with open('./result/proto_test.prototxt', 'w') as f:
        f.write(str(qlstm('val', config.VAL_BATCH_SIZE, \
            config.MAX_CHARS_IN_QUESTION, len(question_char_vocab))))

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
