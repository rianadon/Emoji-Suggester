"""
This implementation of an emoji matcher limits the corpus to all words that are
within a certain degree of an emoji. This allows the program to not have to load
the full dataset, but allows for a greater number of matched emojis to be found.
"""
import json
import os.path
import urllib.request

from gensim import matutils
from gensim.models.keyedvectors import KeyedVectors
import numpy as np

# Word2vec trained model
BIN_NAME = 'GoogleNews-vectors-negative300.bin'
MAX_DEGREE = 0.5 # Not really a degree; just a number from 0 to 1 representing similarity
SAVE_NAME = 'vectors.bin'

# Maximum similarity file
DP_NAME = 'dp.npy'

# Emojilib
EMOJI_URL = 'https://raw.githubusercontent.com/muan/emojilib/master/emojis.json'
EMOJI_NAME = 'emojis.json'

NUM_EMOJIS = 10 # Number of emojis to print
CATEGORY_LENGTH = 8 # Disregard categories with greater or equal to this many elements

if __name__ == '__main__':

    # Download emojilib
    if not os.path.isfile(EMOJI_NAME):
        with open(EMOJI_NAME, 'wb') as f:
            with urllib.request.urlopen(EMOJI_URL) as response:
                f.write(response.read())
        print('Emojilib downloaded!')
    else:
        print('Emojilib already downloaded')

    # Parse emojilib
    with open(EMOJI_NAME, 'r', encoding='utf-8') as f:
        emojis = json.load(f)
    print('Emojilib loaded')

    # For each word in the corpus generate its max similarity to the emoji corpus
    if not os.path.isfile(DP_NAME):

        # Load the word2vec model
        print('Loading model')
        model = KeyedVectors.load_word2vec_format(BIN_NAME, binary=True)
        print('Model loaded!')

        # Create a corpus from emojilib
        wordcorpus = set()
        for name in emojis:
            if name in model.vocab:
                wordcorpus.add(name)
            for keyword in emojis[name]['keywords']:
                if keyword in model.vocab:
                    wordcorpus.add(keyword)
        # Precompute word vectors so the loops are faster
        wcl = list(wordcorpus)
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
        CHUNKSIZE = 1000
        for c in range(0, int(inarr.shape[0]/CHUNKSIZE)):
            cend = min(inarr.shape[0], (c+1)*CHUNKSIZE)
            outarr[c*CHUNKSIZE:cend] = np.amax(np.inner(inarr[c*CHUNKSIZE:cend], corpus), axis=1)

        np.save(DP_NAME, outarr)

        del inarr
        del outarr

    # Now limit the corpus to words over a certain frequency
    if not os.path.isfile(SAVE_NAME):
        # Load the word2vec model
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

     # Load the reduced word2vec model
    print('Loading model')
    model = KeyedVectors.load_word2vec_format(SAVE_NAME, binary=True)
    print('Model loaded!')

    # Create a corpus map from emojilib with each word mapped to its emoji
    corpusmap = {}
    for name in emojis:
        if name in model.vocab:
            if name not in corpusmap:
                corpusmap[name] = []
            corpusmap[name].append(name)
        for keyword in emojis[name]['keywords']:
            if keyword in model.vocab:
                if keyword not in corpusmap:
                    corpusmap[keyword] = []
                corpusmap[keyword].append(name)

    # Precompute word vectors so the loops are faster
    wcl = list(corpusmap.items())
    corpus = np.array([matutils.unitvec(model.word_vec(word)) for word, _ in wcl])
    print('Created corpus with {} elements'.format(len(corpus)))

    # Interactive console
    print('Enter a word to get emojis; type EXIT to stop')
    while True:
        inp = input()
        if inp == 'EXIT':
            break
        try:
            dotprod = np.dot(corpus, matutils.unitvec(model.word_vec(inp)))
            # Find the matches with the most similarity
            matches = np.argpartition(dotprod, -NUM_EMOJIS)[-NUM_EMOJIS:]
            sortedmatches = matches[np.argsort(dotprod[matches])][::-1]

            # First find matching emojis that aren't in large categories
            goodnames = []
            goodcategories = []
            nameset = set()
            for index in sortedmatches:
                names = wcl[index][1]
                if len(names) < CATEGORY_LENGTH:
                    for name in names:
                        if name not in nameset:
                            goodnames.append((name, dotprod[index]))
                            nameset.add(name)
                else:
                    goodcategories.append((names, dotprod[index]))

            # If there aren't enough then start looking at categories
            if len(goodnames) < NUM_EMOJIS:
                goodcategories.sort(key=lambda x: len(x[0]))
                for category, similarity in goodcategories:
                    for name in category:
                        if name not in nameset:
                            goodnames.append((name, similarity))
                            nameset.add(name)
                            if len(goodnames) == NUM_EMOJIS:
                                break
                    if len(goodnames) == NUM_EMOJIS:
                        break

            # Now print the names
            for name, similarity in goodnames[:NUM_EMOJIS]:
                print('{}: {}'.format(emojis[name]['char'], similarity))

        except KeyError:
            print('Sorry: I could not find any good emojis')
