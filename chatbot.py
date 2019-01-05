"""Most of the code comes from seq2seq tutorial. Binary for training conversation models and decoding from them.

Running this program without --decode will  tokenize it in a very basic way,
and then start training a model saving checkpoints to --train_dir.

Running with --decode starts an interactive loop so you can see how
the current checkpoint performs

See the following papers for more information on neural translation models.
 * http://arxiv.org/abs/1409.3215
 * http://arxiv.org/abs/1409.0473
 * http://arxiv.org/abs/1412.2007
"""

import math
import sys
import time
from data_preprocessing import *
from model import *
from tqdm import tqdm

tf.app.flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
tf.app.flags.DEFINE_integer("batch_size", 256, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("numEpochs", 30, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("size", 512, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 3, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("en_vocab_size", 40000, "English vocabulary size.")
tf.app.flags.DEFINE_integer("en_de_seq_len", 20, "English vocabulary size.")
tf.app.flags.DEFINE_integer("max_train_data_size", 0, "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 100, "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_string("train_dir", './tmp', "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_integer("beam_size", 5, "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_boolean("beam_search", True, "Set to True for beam_search.")
tf.app.flags.DEFINE_boolean("decode", True, "Set to True for interactive decoding.")
FLAGS = tf.app.flags.FLAGS

def create_model(session, forward_only, beam_search, beam_size = 5):
    """Create translation model and initialize or load parameters in session."""
    model = Seq2SeqModel(
        FLAGS.en_vocab_size, FLAGS.en_vocab_size, [10, 10],
        FLAGS.size, FLAGS.num_layers, FLAGS.batch_size,
        FLAGS.learning_rate, forward_only=forward_only, beam_search=beam_search, beam_size=beam_size)
    ckpt = tf.train.latest_checkpoint(FLAGS.train_dir)
    model_path = '/Users/cassini/PycharmProjects/blackbox/tmp/chat_bot.ckpt-0'
    if forward_only:
        model.saver.restore(session, model_path)
    elif ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print("Created model with fresh parameters.")
        session.run(tf.initialize_all_variables())
    return model

def train():
    # prepare dataset
    data_path = '/Users/cassini/PycharmProjects/blackbox/data/cornell_xwt_model.pkl'
    word2id, id2word, trainingSamples = loadData(data_path)
    with tf.Session() as sess:
        print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))

        # train the model
        model = create_model(sess, False, beam_search=False, beam_size=5)


        current_step = 0
        for e in range(FLAGS.numEpochs):
            print("----- Epoch {}/{} -----".format(e + 1, FLAGS.numEpochs))
            batches = getBatch(trainingSamples, FLAGS.batch_size, model.en_de_seq_len)
            for nextBatch in tqdm(batches, desc="Training"):
                _, step_loss = model.step(sess, nextBatch.encoderSeqs, nextBatch.decoderSeqs, nextBatch.targetSeqs,
                                          nextBatch.weights, go_Token)
                current_step += 1
                if current_step % FLAGS.steps_per_checkpoint == 0:
                    perplexity = math.exp(float(step_loss)) if step_loss < 300 else float('inf')
                    tqdm.write("----- Step %d -- Loss %.2f -- Perplexity %.2f" % (current_step, step_loss, perplexity))

                    # save the checkpoint
                    checkpoint_path = os.path.join(FLAGS.train_dir, "chat_bot.ckpt")
                    model.saver.save(sess, checkpoint_path, global_step=model.global_step)

def decode(read_sentence):
    with tf.Session() as sess:
        beam_size = FLAGS.beam_size
        beam_search = FLAGS.beam_search
        # create model
        model = create_model(sess, True, beam_search=beam_search, beam_size=beam_size)

        # batch_size = 1
        model.batch_size = 1
        data_path = '/Users/cassini/PycharmProjects/blackbox/data/dataset-cornell-length10-filter1-vocabSize40000.pkl'

        # load word_dictonary
        word2id, id2word, trainingSamples = loadData(data_path)

        if beam_search:
            sys.stdout.write("please input:")
            sys.stdout.flush()

            # read from command line
            sentence = read_sentence

            batch = sentence_to_encoder(sentence, word2id, model.en_de_seq_len)

            # check the sentence_sequence
            print("batch.encoder_Sequence")
            print(batch.encoder_Sequence)

            print("batch.decoder_Sequence")
            print(batch.decoder_Sequence)

            print("batch.target_Sequence")
            print(batch.target_Sequence)

            # using model.step function
            beam_path, beam_symbol = model.step(sess, batch.encoder_Sequence, batch.decoder_Sequence, batch.target_Sequence,
                                                    batch.weights, go_Token)
            print("beam_path")
            print(beam_path)
            print("beam_symbol")
            print(beam_symbol)
            paths = [[] for _ in range(beam_size)]
            #[[],[],[]]
            curr = [i for i in range(beam_size)]
            #[0,1,2,3...]
            num_steps = len(beam_path)

            # reversed order
            for i in range(num_steps-1, -1, -1):
                for kk in range(beam_size):
                    # beam_symbol
                    paths[kk].append(beam_symbol[i][curr[kk]])
                    print("paths")
                    print(paths)
                    curr[kk] = beam_path[i][curr[kk]]
            recos = set()
            print("chatbot said:")
            rec_set = []
            for kk in range(beam_size):
                foutputs = [int(logit) for logit in paths[kk][::-1]]
                compat = []
                if eos_Token in foutputs:
                    foutputs = foutputs[:foutputs.index(eos_Token)]
                    for output in foutputs:
                        if output in id2word:
                            print("tf.compat.as_str(id2word[output])n")
                            print(tf.compat.as_str(id2word[output]))
                            compat.append(tf.compat.as_str(id2word[output]))
                            print("compat:")
                            print(compat)
                rec = " ".join(compat)
                rec_set.append(rec)


                    #rec = " ".join([tf.compat.as_str(id2word[output]) for output in foutputs if output in id2word])
                if rec not in recos:
                    recos.add(rec)
                    print(rec)
            return rec_set

            # with sentence alive
            # while sentence:
            #     # one batch only contained one sentence
            #     batch = sentence_to_encoder(sentence, word2id, model.en_de_seq_len)

            #     # check the sentence_sequence
            #     print("batch.encoder_Sequence")
            #     print(batch.encoder_Sequence)

            #     print("batch.decoder_Sequence")
            #     print(batch.decoder_Sequence)

            #     print("batch.target_Sequence")
            #     print(batch.target_Sequence)

            #     # using model.step function
            #     beam_path, beam_symbol = model.step(sess, batch.encoder_Sequence, batch.decoder_Sequence, batch.target_Sequence,
            #                                         batch.weights, go_Token)
            #     print("beam_path")
            #     print(beam_path)
            #     print("beam_symbol")
            #     print(beam_symbol)
            #     paths = [[] for _ in range(beam_size)]
            #     #[[],[],[]]
            #     curr = [i for i in range(beam_size)]
            #     #[0,1,2,3...]
            #     num_steps = len(beam_path)

            #     # reversed order
            #     for i in range(num_steps-1, -1, -1):
            #         for kk in range(beam_size):
            #             # beam_symbol
            #             paths[kk].append(beam_symbol[i][curr[kk]])
            #             print("paths")
            #             print(paths)
            #             curr[kk] = beam_path[i][curr[kk]]
            #     recos = set()
            #     print("chatbot said:")
            #     for kk in range(beam_size):
            #         foutputs = [int(logit) for logit in paths[kk][::-1]]
            #         compat = []
            #         if eos_Token in foutputs:
            #             foutputs = foutputs[:foutputs.index(eos_Token)]
            #             for output in foutputs:
            #                 if output in id2word:
            #                     print("tf.compat.as_str(id2word[output])n")
            #                     print(tf.compat.as_str(id2word[output]))
            #                     compat.append(tf.compat.as_str(id2word[output]))
            #                     print("compat:")
            #                     print(compat)
            #         rec = " ".join(compat)

            #         #rec = " ".join([tf.compat.as_str(id2word[output]) for output in foutputs if output in id2word])
            #         if rec not in recos:
            #             recos.add(rec)
            #             print(rec)
            #     print("please input: ", "")
            #     sys.stdout.flush()
            #     sentence = sys.stdin.readline()

# def main(_):
#   if FLAGS.decode:
#     decode()
#   else:
#     train()

# if __name__ == "__main__":
#   tf.app.run()
