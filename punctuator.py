# coding: utf-8
from __future__ import division

import models, data, main, process_text

import sys
import codecs
import re

import tensorflow as tf
import numpy as np

MAX_SUBSEQUENCE_LEN = 200

def to_array(arr, dtype=np.int32):
    # minibatch of 1 sequence as column
    return np.array([arr], dtype=dtype).T

def makeReadable(original, tkspredicted):
    res = ""
    capitalize = False
    numId = 0
    nums = re.findall('\d+', original)
    last_is_punct = False

    for tk in tkspredicted:
        if tk == process_text.NUM:
            res += ' ' + nums.pop(0)
            capitalize = False
            last_is_punct = False

        elif tk in data.PUNCTUATION_VOCABULARY:
            last_is_punct = True
            capitalize = False

            if tk[-1] == 'M':
              capitalize = True

            if tk == '·M' or tk == data.SPACE:
                res += ' '
            elif tk != process_text.BREAK:
                res += tk[0] + ' '


        else:
          res += ('' if last_is_punct else ' ') + (tk[0].upper() if capitalize else tk[0]) + tk[1:]
          capitalize = False
          last_is_punct = False
  
    return re.sub('\s+', ' ', res).strip()

def restoreLine(text, word_vocabulary, reverse_punctuation_vocabulary, model):
    res = []
    i = 0
    while True:
        subsequence = text[i:i+MAX_SUBSEQUENCE_LEN]

        if len(subsequence) == 0:
            break

        converted_subsequence = [word_vocabulary.get(w, word_vocabulary[data.UNK]) for w in subsequence]

        y = predict(to_array(converted_subsequence), model)

        res.append(subsequence[0])

        last_eos_idx = 0
        punctuations = []
        for y_t in y:

            p_i = np.argmax(tf.reshape(y_t, [-1]))
            punctuation = reverse_punctuation_vocabulary[p_i]

            punctuations.append(punctuation)

            if punctuation in data.EOS_TOKENS:
                last_eos_idx = len(punctuations) # we intentionally want the index of next element

        if subsequence[-1] == process_text.END:
            step = len(subsequence) - 1
            
        elif last_eos_idx != 0:
            step = last_eos_idx
        else:
            step = len(subsequence) - 1

        for j in range(step):
            res += [punctuations[j]]
            if j < step - 1:
                res.append(subsequence[1+j])

        if subsequence[-1] == process_text.END:
            res.append(subsequence[-1])
            break

        i += step
   
    return res[1:-1] 

def restore(output_file, text, word_vocabulary, reverse_punctuation_vocabulary, model):
    i = 0
    with open(output_file, 'w') as f_out:
        while True:

            subsequence = text[i:i+MAX_SUBSEQUENCE_LEN]

            if len(subsequence) == 0:
                break

            converted_subsequence = [word_vocabulary.get(w, word_vocabulary[data.UNK]) for w in subsequence]

            y = predict(to_array(converted_subsequence), model)

            f_out.write(subsequence[0])

            last_eos_idx = 0
            punctuations = []
            for y_t in y:

                p_i = np.argmax(tf.reshape(y_t, [-1]))
                punctuation = reverse_punctuation_vocabulary[p_i]

                punctuations.append(punctuation)

                if punctuation in data.EOS_TOKENS:
                    last_eos_idx = len(punctuations) # we intentionally want the index of next element

            if subsequence[-1] == process_text.END:
                step = len(subsequence) - 1
            elif last_eos_idx != 0:
                step = last_eos_idx
            else:
                step = len(subsequence) - 1

            for j in range(step):
                f_out.write(" " + punctuations[j] + " " if punctuations[j] != data.SPACE else " ")
                if j < step - 1:
                    f_out.write(subsequence[1+j])

            if subsequence[-1] == process_text.END:
                break

            i += step

def predict(x, model):
    return tf.nn.softmax(net(x))

if __name__ == "__main__":
    continuous_text = True

    if len(sys.argv) > 1:
        model_file = sys.argv[1]
    else:
        sys.exit("Model file path argument missing")

    if len(sys.argv) > 2:
        input_file = sys.argv[2]
    else:
        sys.exit("Input file path argument missing")

    if len(sys.argv) > 3:
        output_file = sys.argv[3]
    else:
        sys.exit("Output file path argument missing")

    if len(sys.argv) > 4 and sys.argv[4]=='--bylines':
        continuous_text = False
        print("Processing text line by line")
    else:
        print("Assuming continuous text")

    vocab_len = len(data.read_vocabulary(data.WORD_VOCAB_FILE))
    x_len = vocab_len if vocab_len < data.MAX_WORD_VOCABULARY_SIZE else data.MAX_WORD_VOCABULARY_SIZE + data.MIN_WORD_COUNT_IN_VOCAB
    x = np.ones((x_len, main.MINIBATCH_SIZE)).astype(int)

    print("Loading model parameters...")
    net, _ = models.load(model_file, x)

    word_vocabulary = net.x_vocabulary
    punctuation_vocabulary = net.y_vocabulary

    reverse_word_vocabulary = {v:k for k,v in word_vocabulary.items()}
    reverse_punctuation_vocabulary = {v:k for k,v in punctuation_vocabulary.items()}

    print('Restoring punctuation...')

    if continuous_text:
        with open(input_file, 'r') as f:
            input_text = f.read()
            input_text = process_text.process_line(input_text.strip())
      
        if len(input_text) == 0:
            sys.exit("Input file empty.")

        text = [w for w in input_text.split() if w not in punctuation_vocabulary and w not in data.PUNCTUATION_MAPPING] 

        restore(output_file, text, word_vocabulary, reverse_punctuation_vocabulary, net)

        print("Finished! Inspect %s for the result" % output_file)
    
    else:
        li = 0
        with open(input_file, 'r') as f:
            line = f.readline()
            while line:
              input_line = process_text.process_line(line.strip())
              input_line = [w for w in input_line.split() if w not in punctuation_vocabulary and w not in data.PUNCTUATION_MAPPING]
              punct_line = restoreLine(input_line, word_vocabulary, reverse_punctuation_vocabulary, net)
              #print('processed line %d' % li)

              with open(output_file, ('a' if li else 'w')) as f_out:
                  readable_punct_line = makeReadable(line, punct_line)
                  #print(readable_punct_line)
                  f_out.write(readable_punct_line + '\n')

              line = f.readline()
              li += 1

            if li == 0:
                sys.exit("Input file empty.")
            
            else:
                print("Finished! Inspect %s for the result" % output_file)
              


