import tensorflow as tf
import modekeys
from tensorflow.python.layers import core as layers_core
import helper

random_seed = 17

def create_word_embedding_matrix(word_dim):
    vocab,vocab_dict = helper.load_vocab('./twitter_data/rg_vocab.txt')
    glove_vectors,glove_dict = helper.load_glove_vectors('./twitter_data/my_vector.txt',vocab)
    initial_value = helper.build_initial_embedding_matrix(vocab_dict,glove_dict,glove_vectors,word_dim)
    embedding_w = tf.get_variable(name='embedding_W',initializer=initial_value,trainable=True)
    return embedding_w

def model_impl(query, response_in, response_out, response_mask, query_length,hp,mode):
    debug_tensors = []

    with tf.variable_scope('embedding_layer') as vs:
        embedding_W = create_word_embedding_matrix(hp.word_dim)
        query = tf.nn.embedding_lookup(embedding_W,query,name='query_embedding')
        if mode == modekeys.TRAIN or mode == modekeys.EVAL:
            response_in = tf.nn.embedding_lookup(embedding_W,response_in,name='response_in_embedding')

    with tf.variable_scope('encoder') as vs:
        with tf.variable_scope('fw') as vs:
            kernek_initializer = tf.random_uniform_initializer(minval= -0.1,maxval=0.1,seed=random_seed)
            bias_initializer = tf.zeros_initializer()
            encoder_fw_cell = tf.nn.rnn_cell.GRUCell(num_units=hp.rnn_num_units,kernel_initializer=kernek_initializer,bias_initializer=bias_initializer) #must use initializer op not constant op because itself will decide the shape
        with tf.variable_scope('bw') as vs:
            kernek_initializer = tf.random_uniform_initializer(minval=-0.1, maxval=0.1, seed=random_seed)
            bias_initializer = tf.zeros_initializer()
            encoder_bw_cell = tf.nn.rnn_cell.GRUCell(num_units=hp.rnn_num_units, kernel_initializer=kernek_initializer,
                                                  bias_initializer=bias_initializer)
        query_length = tf.squeeze(query_length,axis=1,name='squeeze_query_length')
        query_hidden_states, query_encoding = tf.nn.bidirectional_dynamic_rnn(encoder_fw_cell,encoder_bw_cell,query,sequence_length=query_length,initial_state_fw=encoder_fw_cell.zero_state(hp.batch_size,tf.float32),initial_state_bw=encoder_bw_cell.zero_state(hp.batch_size,tf.float32),swap_memory=True)
        query_hidden_states = tf.concat(query_hidden_states,axis=2)


    with tf.variable_scope('decoder') as vs:
        #helper, atten_mechan, atten_wrapper
        if mode == modekeys.TRAIN:
            sequence_length = tf.constant(value=hp.max_sentence_length,dtype=tf.int32,shape=[hp.batch_size],name='seq_length_train_helper')
            helper = tf.contrib.seq2seq.TrainingHelper(inputs=response_in,sequence_length=sequence_length) #sequence_length should be the max_size as input
        elif mode == modekeys.EVAL:
            sequence_length = tf.constant(value=hp.max_sentence_length, dtype=tf.int32, shape=[hp.eval_batch_size])
            helper = tf.contrib.seq2seq.TrainingHelper(inputs=response_in,sequence_length=sequence_length)
        else:
            start_tokens = tf.constant(value=1, dtype=tf.int32, shape=[hp.eval_batch_size], name='start_tokens')
            end_token = 1
            helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding=embedding_W, start_tokens=start_tokens,
                                                              end_token=end_token)  # later change to beam search

        kernek_initializer = tf.random_uniform_initializer(minval=-0.1, maxval=0.1, seed=random_seed)
        bias_initializer = tf.zeros_initializer()
        decoder_cell = tf.nn.rnn_cell.GRUCell(num_units=hp.decoder_rnn_num_units,kernel_initializer=kernek_initializer,bias_initializer=bias_initializer)
        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                num_units=hp.decoder_rnn_num_units, memory=query_hidden_states,
                memory_sequence_length=query_length) #num_units is the num units in rnn cell
        attn_cell = tf.contrib.seq2seq.AttentionWrapper(
            decoder_cell, attention_mechanism, attention_layer_size=None,output_attention=False)
        output_layer = layers_core.Dense(units=hp.vocab_size,
                                       activation=None,
                                       use_bias=False) # should use no activation and no bias

        if mode == modekeys.TRAIN:
            decoder = tf.contrib.seq2seq.BasicDecoder(cell=attn_cell,
                                                      helper=helper,
                                                      initial_state=attn_cell.zero_state(batch_size=hp.batch_size,
                                                                                         dtype=tf.float32),
                                                      output_layer=output_layer)
            final_outputs, final_state, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(decoder=decoder,swap_memory=True)
            logit = final_outputs.rnn_output #[batch_size, max_sentence_size, vocab_size]
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=response_out,logits=logit)
            loss = tf.multiply(cross_entropy,tf.to_float(response_mask))
            loss = tf.reduce_sum(loss,axis=1)
            loss = tf.reduce_mean(loss)
            debug_tensors.append(logit)
            # loss = sparse_softmax_cross_entropy_with_value_clip(logit,response_out,hp.vocab_size,response_mask,debug_tensors)
            return loss, debug_tensors
        elif mode == modekeys.EVAL:
            decoder = tf.contrib.seq2seq.BasicDecoder(cell=attn_cell,
                                                      helper=helper,
                                                      initial_state=attn_cell.zero_state(batch_size=hp.eval_batch_size,
                                                                                         dtype=tf.float32),
                                                      output_layer=output_layer)
            final_outputs, final_state, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(decoder=decoder,swap_memory=True,impute_finished = True) #if use impute_finished, don't need response mask
            return final_outputs.sample_id, final_sequence_lengths, final_outputs.rnn_output # [batch_size], each entry is coresponding decoded length
        elif mode == modekeys.PREDICT:
            decoder = tf.contrib.seq2seq.BasicDecoder(cell=attn_cell,
                                                      helper=helper,
                                                      initial_state=attn_cell.zero_state(batch_size=hp.eval_batch_size,
                                                                                         dtype=tf.float32),
                                                      output_layer=output_layer)
            final_outputs,final_state,final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(decoder=decoder, swap_memory=True, impute_finished=True)
            return final_outputs.sample_id, final_sequence_lengths,final_outputs.rnn_output

def sparse_softmax_cross_entropy_with_value_clip(logit,labels,depth,weights,debugs):
    softmax = tf.clip_by_value(t=tf.nn.softmax(logits=logit), clip_value_max=0.99, clip_value_min=0.00001)
    one_hot_label = tf.one_hot(indices=labels, depth=depth, on_value=1.0, off_value=0.0, axis=-1,
                               dtype=tf.float32)
    log = tf.log(softmax)
    loss = tf.multiply(log, one_hot_label)
    loss = -tf.reduce_sum(loss, axis=[2])
    loss = tf.multiply(loss,tf.to_float(weights))
    loss = tf.reduce_mean(tf.reduce_sum(loss,axis=1))
    return loss

