import tensorflow as tf
import input_layer
import modekeys
import hparam
import encoder_decoder
from tensorflow.python.training import saver as saver_lib
import bleu
import numpy as np
from tensorflow.core.framework import summary_pb2
import os



def evaluate(eval_file,model_dir,summary_dir,train_steps):
    hp = hparam.create_hparam()

    eval_graph = tf.Graph()
    with eval_graph.as_default():
        if train_steps is not None:
            mode = modekeys.EVAL
        else:
            mode = modekeys.PREDICT
        features = input_layer.create_input_layer(mode=mode, filenames=[eval_file],
                                                  batch_size=hp.eval_batch_size,
                                                  num_epochs=1, shuffle_batch=False,
                                                  max_sentence_length=hp.max_sentence_length)
        query, response_in, response_out, response_mask, query_length = features
        sample_ids, final_lengths,logits = encoder_decoder.model_impl(query=query,
                                   response_in=response_in,
                                   response_out=response_out,
                                   response_mask=response_mask,
                                   query_length=query_length,
                                   hp=hp,
                                   mode=mode)

        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=response_out, logits=logits)
        ppl = tf.reduce_mean(tf.multiply(cross_entropy, tf.to_float(response_mask)))

        # wrong_predictions = tf.not_equal(tf.to_int64(sample_ids),tf.to_int64(response_out))
        # wrong_predictions = tf.multiply(tf.to_float(response_mask),tf.cast(wrong_predictions, tf.float32))
        # WER = tf.reduce_sum(tf.cast(wrong_predictions, tf.float32))
        WER,wer_update_op = tf.metrics.accuracy(labels=response_out,predictions=sample_ids,weights=response_mask)

        sess = tf.Session()

        saver = tf.train.Saver()
        checkpoint = saver_lib.latest_checkpoint(model_dir)
        saver.restore(sess=sess,save_path=checkpoint)
        sess.run(tf.local_variables_initializer())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess,coord=coord)
        tf.logging.info('Begin evaluation at model {} on file {}'.format(checkpoint,eval_file))
        total_bleu_score = 0
        total_ppl = 0
        eval_step = 0
        try:
            while not coord.should_stop():
                if train_steps is not None:
                    gen_ids, ref_ids,gen_lengths,perplexity,_ = sess.run(fetches=[sample_ids, response_out,final_lengths,ppl,wer_update_op])
                    score = calculate_bleu_score(generate_response=gen_ids,reference_response=ref_ids)
                    total_bleu_score += score
                    total_ppl += perplexity
                else:
                    query_ids, gen_ids, gen_lengths, ref_ids = sess.run(fetches=[query,sample_ids,final_lengths,response_out])
                    print('write to file')
                    write_to_file(query_ids,ref_ids,gen_ids,'./data')
                    coord.request_stop()

                eval_step += 1
        except tf.errors.OutOfRangeError:
            word_error_rate = sess.run(WER) # final run to get the final WER value
            tf.logging.info('Finish evaluation')
        finally:
            coord.request_stop()
        coord.join(threads)

        if train_steps:
            bleu_score = total_bleu_score / eval_step
            avg_ppl = total_ppl /eval_step
            avg_wer = word_error_rate or 0
            write_to_summary(output_dir=summary_dir,summary_tag='eval_bleu_score',summary_value=bleu_score,current_global_step=train_steps)
            write_to_summary(output_dir=summary_dir,summary_tag='eval_ppl',summary_value=avg_ppl,current_global_step=train_steps)
            write_to_summary(output_dir=summary_dir,summary_tag='eval_word_error_rate',summary_value=avg_wer,current_global_step=train_steps)
            tf.logging.info('ppl is {}'.format(avg_ppl))
            tf.logging.info('bleu score is {}'.format(bleu_score))
            tf.logging.info('word error rate is {}'.format(avg_wer))

        return ppl

def calculate_bleu_score(generate_response, reference_response):
    #reference_corpus is like [[[token1, token2, token3]]]
    reference_corpus = [[ref.tolist()] for ref in reference_response]
    #translation corpus is like [[token1, token2]]
    translation_corpus = [gen.tolist() for gen in generate_response]
    result = bleu.compute_bleu(reference_corpus=reference_corpus,translation_corpus=translation_corpus)
    return result[0]

def write_to_summary(output_dir,summary_tag,summary_value,current_global_step):
    summary_writer = tf.summary.FileWriterCache.get(output_dir)
    summary_proto = summary_pb2.Summary()
    value = summary_proto.value.add()
    value.tag = summary_tag
    if isinstance(summary_value, np.float32) or isinstance(summary_value, float):
        value.simple_value = float(summary_value)
    elif isinstance(summary_value,int) or isinstance(summary_value, np.int64) or isinstance(summary_value, np.int32):
        value.simple_value = int(summary_value)
    summary_writer.add_summary(summary_proto, current_global_step)
    summary_writer.flush()

def write_to_file(query,response,generations,data_dir):
    vocabulary = load_vocabulary(os.path.join(data_dir,'vocabulary.txt'))
    filepath = os.path.join(data_dir,'generate_response.txt')
    with open(filepath,'w') as f:
        for q,r,gen in zip(query,response,generations):
            if len(set(gen)) >3 :
                q_words = replace_to_words(q,vocabulary)
                r_words = replace_to_words(r,vocabulary)
                gen_words = replace_to_words(gen,vocabulary)
                f.write(' '.join(q_words) + '|||\n')
                f.write(' '.join(r_words) + '|||\n')
                f.write(' '.join(gen_words) + '|||\n\n')

def replace_to_words(ids,vocab):
    return [vocab[i] for i in ids]

def load_vocabulary(vocab_path):
    vocabulary = {}
    with open(vocab_path, 'r') as f:
        for i,l in enumerate(f.readlines()): # unk index = 0 eos index = 1
            vocabulary[i] = l.rstrip('\n')
    return vocabulary

if __name__ == '__main__':
    evaluate('./data/validation.tfrecords','./model/model3','./model/model3/summary/eval_test',None)


