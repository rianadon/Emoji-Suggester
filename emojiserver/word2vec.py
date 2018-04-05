"""
This module gives access to the limitted word2vec model that contains the words
similar to the emojis in emojilib.
"""

import gc
import os

import numpy as np
from gensim import matutils
from gensim.models.keyedvectors import KeyedVectors

from paths import BIN_NAME, DP_NAME, SAVE_NAME, NSAVE_NAME

MAX_DEGREE = 0.5 # Not really a degree; just a number from 0 to 1 representing similarity
CHUNKSIZE = 1000 # For splitting up memmaps

def generate_dps(wordcorpus):
    """Generate the maximum similarity of each word to emoji names.

    It takes in a corpus which is a set of all emoji words.
    """

    print('Loading model')
    model = KeyedVectors.load_word2vec_format(BIN_NAME, binary=True)
    print('Model loaded!')

    # Limit the corpus to words in the model
    wcl = list(word for word in wordcorpus if word in model.vocab)

    # Precompute word vectors so the loops are faster
    corpus = np.array([matutils.unitvec(model.word_vec(word)) for word in wcl])
    print('Created corpus with {} elements'.format(len(corpus)))

    print('Computing norms')
    model.init_sims(replace=True)

    # Save memory by deleting non-normed data
    syn0norm = model.syn0norm
    del model

    # Convert sys0norm to a memmap to further reduce memory
    print('Saving to memmap')
    inarr = np.memmap('inmemmap.dat', dtype=syn0norm.dtype, mode='w+', shape=syn0norm.shape)
    inarr[:] = syn0norm[:]
    outarr = np.memmap('outmemmap.dat', dtype=syn0norm.dtype, mode='w+', shape=(syn0norm.shape[0],))

    # Discard the array now that it's stored in a memmap
    del syn0norm

    print('Computing dot products')
    for c in range(0, int(inarr.shape[0]/CHUNKSIZE)):
        cend = min(inarr.shape[0], (c+1)*CHUNKSIZE)
        outarr[c*CHUNKSIZE:cend] = np.amax(np.inner(inarr[c*CHUNKSIZE:cend], corpus), axis=1)

    np.save(DP_NAME, outarr)

    del inarr
    del outarr
    gc.collect()

    os.remove('inmemmap.dat')
    os.remove('outmemmap.dat')

def generate_limittedmodel():
    """Generate the word2vec model with a subset of the original vocab.

    The dot products will need to have been computed, so `generate_dps()` may
    need to be called before this function.
    """
    print('Loading model')
    model = KeyedVectors.load_word2vec_format(BIN_NAME, binary=True)
    print('Model loaded!')

    print('Loading dot products')
    dp = np.load(DP_NAME)
    print('Dot products loaded')

    print('Filtering vocab')
    for name, vocab in list(model.vocab.items()):
        if dp[vocab.index] < MAX_DEGREE:
            del model.vocab[name]

    il = list(model.vocab.items())
    print('Sorting vocab')
    il.sort(key=lambda x: x[1].index)

    # Find the indexes of the words that are being kept
    print('Generating indexes')
    indexes = []
    for i in range(0, len(il)):
        name, vocab = il[i]
        indexes.append(vocab.index)
        model.vocab[name].index = i

    print('Modifying model weights')
    model.syn0 = model.syn0[indexes]

    print('Saving file')
    model.save_word2vec_format(SAVE_NAME, binary=True)

def generate_normedmodel():
    """Generate a word2vec model with all vectors normed."""
    # Load the reduced word2vec model
    print('Loading model')
    model = KeyedVectors.load_word2vec_format(SAVE_NAME, binary=True)
    print('Model loaded!')

    print('Computing norms')
    model.init_sims(replace=True)

    print('Saving model')
    model.save(NSAVE_NAME)

def normedmodel(corpus):
    """Return the limitted word2vec model.

    The function takes in a corpus which is a set of all emoji words.
    """
    if not os.path.isfile(NSAVE_NAME):
        if not os.path.isfile(SAVE_NAME):
            if not os.path.isfile(DP_NAME):
                generate_dps(corpus)
            generate_limittedmodel()
        generate_normedmodel()

    # Load the reduced word2vec model
    print('Loading model')
    model = KeyedVectors.load(NSAVE_NAME, mmap='r')
    print('Model loaded!')

    return model

def vectorcorpus(model, wcl):
    """Return an array of word vectors for the dict of words.

    The provided model is assumed to have normed vectors.
    """
    corpus = np.array([model.word_vec(word) for word, _ in wcl])
    print('Created corpus with {} elements'.format(len(corpus)))
    return corpus
