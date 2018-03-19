import tensorflow as tf
import modekeys

def create_input_layer(mode,filenames,num_epochs,batch_size,shuffle_batch,max_sentence_length):
    with tf.name_scope('input_layer') as ns:
        example = read_and_decode(filenames, num_epochs, max_sentence_length)
        min_after_dequeue = 10000
        capacity = min_after_dequeue + 3 * batch_size
        if shuffle_batch:
            batch_example = tf.train.shuffle_batch(example,batch_size=batch_size,
                                                   capacity=capacity,min_after_dequeue=min_after_dequeue)
        else:
            batch_example = tf.train.batch(example,batch_size=batch_size)

        query = batch_example.pop('query')
        response_in = batch_example.pop('response_in')
        response_out = batch_example.pop('response_out')
        response_mask = batch_example.pop('response_mask')
        query_length = batch_example.pop('query_length')

        if mode == modekeys.TRAIN or mode == modekeys.EVAL:
            return query,response_in,response_out,response_mask,query_length
        elif mode == modekeys.PREDICT:
            return query,response_in,response_out,response_mask,query_length

def read_and_decode(filenames,num_epochs,max_sentence_length):
    fname_queue = tf.train.string_input_producer(filenames,num_epochs=num_epochs)
    reader = tf.TFRecordReader("my_reader")
    _, serilized_example = reader.read(queue=fname_queue)
    feature_spec = create_feature_spec(max_sentence_length)
    example = tf.parse_single_example(serilized_example, feature_spec)
    return example

def create_feature_spec(max_sentence_length):
    spec = {}
    spec['query'] = tf.FixedLenFeature(shape=[max_sentence_length],dtype=tf.int64)
    spec['response_in'] = tf.FixedLenFeature(shape=[max_sentence_length], dtype=tf.int64)
    spec['response_out'] = tf.FixedLenFeature(shape=[max_sentence_length], dtype=tf.int64)
    spec['response_mask'] = tf.FixedLenFeature(shape=[max_sentence_length], dtype=tf.int64)
    spec['query_length'] = tf.FixedLenFeature(shape=[1], dtype=tf.int64)
    return spec