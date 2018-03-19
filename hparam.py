import tensorflow as tf
from collections import namedtuple
import os

# Model Parameters
tf.flags.DEFINE_integer('vocab_size',41834,'vocab size')
tf.flags.DEFINE_integer("word_dim", 200, "Dimensionality of the embeddings")
tf.flags.DEFINE_integer('rnn_num_units', 512, 'Num of rnn cells')
tf.flags.DEFINE_integer('decoder_rnn_num_units', 1024, 'Num of rnn cells')
tf.flags.DEFINE_float('keep_prob', 1.0, 'the keep prob of rnn state')
tf.flags.DEFINE_string('rnn_cell_type', 'GRU', 'the cell type in rnn')

# Pre-trained parameters
tf.flags.DEFINE_integer('max_sentence_length', 30,'the max sentence length')

# Training Parameters, 800845 42598 8520
tf.flags.DEFINE_integer("batch_size", 64, "Batch size during training")
tf.flags.DEFINE_integer("eval_batch_size", 64, "Batch size during evaluation")
tf.flags.DEFINE_integer('num_epochs', 3, 'the number of epochs')
tf.flags.DEFINE_integer('eval_step', 12500, 'eval every n steps')
tf.flags.DEFINE_boolean('shuffle_batch',True, 'whether shuffle the train examples when batch')
tf.flags.DEFINE_float("learning_rate", 0.01, "Learning rate")
tf.flags.DEFINE_integer('summary_save_steps',1000,'steps to save summary')

FLAGS = tf.flags.FLAGS

HParams = namedtuple(
  "HParams",
  [ "eval_step",
    "batch_size",
    "word_dim",
    "eval_batch_size",
    "learning_rate",
    'vocab_size',
    "num_epochs",
    'rnn_num_units',
    'decoder_rnn_num_units',
    'keep_prob',
    'rnn_cell_type',
    'max_sentence_length',
    'shuffle_batch',
    'summary_save_steps'
  ])

def create_hparam():
  return HParams(
    eval_step = FLAGS.eval_step,
    batch_size=FLAGS.batch_size,
    eval_batch_size=FLAGS.eval_batch_size,
    learning_rate=FLAGS.learning_rate,
    word_dim=FLAGS.word_dim,
    vocab_size=FLAGS.vocab_size,
    num_epochs=FLAGS.num_epochs,
    rnn_num_units=FLAGS.rnn_num_units,
      decoder_rnn_num_units=FLAGS.decoder_rnn_num_units,
    keep_prob=FLAGS.keep_prob,
    rnn_cell_type=FLAGS.rnn_cell_type,
    max_sentence_length=FLAGS.max_sentence_length,
    shuffle_batch = FLAGS.shuffle_batch,
    summary_save_steps = FLAGS.summary_save_steps
  )

def write_hparams_to_file(hp, model_dir):
  with open(os.path.join(os.path.abspath(model_dir),'hyper_parameters.txt'), 'w') as f:
    f.write('batch_size: {}\n'.format(hp.batch_size))
    f.write('learning_rate: {}\n'.format(hp.learning_rate))
    f.write('num_epochs: {}\n'.format(hp.num_epochs))
    f.write('rnn_num_units: {}\n'.format(hp.rnn_num_units))
    f.write('keep_prob: {}\n'.format(hp.keep_prob))
