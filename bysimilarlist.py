"""
This implementation of an emoji matcher creates a dictionary of the 100 closest
words for each emoji. It then creates a word -> emoji map from this and operates
from this map. While this does limit the number of words that can be matched to
emojis, it does have the advantage that the generated reverse map is quite
small.
"""

import json
import os.path
import pickle
import urllib.request

# Word2vec trained model
BIN_NAME = 'GoogleNews-vectors-negative300.bin'
SIMILAR_N = 100

# Emojilib
EMOJI_URL = 'https://raw.githubusercontent.com/muan/emojilib/master/emojis.json'
EMOJI_NAME = 'emojis.json'

# Word -> similar word mapping
SIMILARS_NAME = 'similars.pickle'

# Similar -> emojis mapping
REVERSE_NAME = 'reverse.pickle'

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

    # Find similar words
    if not os.path.isfile(SIMILARS_NAME):
        # Create a corpus from emojilib
        corpus = set()
        for name in emojis:
            corpus.add(name)
            for keyword in emojis[name]['keywords']:
                corpus.add(keyword)
        print('Created corpus with {} elemens'.format(len(corpus)))

        # Load the word2vec model
        from gensim.models.keyedvectors import KeyedVectors

        model = KeyedVectors.load_word2vec_format(BIN_NAME, binary=True)
        print('Model loaded!')

        similars = {}
        for word in corpus:
            print('\t' + word)
            try:
                similars[word] = model.similar_by_word(word, topn=SIMILAR_N)
            except KeyError:
                print('\twarning: ' + word + ' not in corpus')
        print('Generated similar words')

        del model

        with open(SIMILARS_NAME, 'wb') as f:
            pickle.dump(similars, f)
        print('Saved similar words')
    else:
        print('Similar words already generated')

    with open(SIMILARS_NAME, 'rb') as f:
        similars = pickle.load(f)


    if not os.path.isfile(REVERSE_NAME):
        # Create a corpus map
        corpusmap = {}
        for name in emojis:
            if name not in corpusmap:
                corpusmap[name] = []
            corpusmap[name].append(name)
            for keyword in emojis[name]['keywords']:
                if keyword not in corpusmap:
                    corpusmap[keyword] = []
                corpusmap[keyword].append(name)
        print('Created corpus map')

        # Map the words in reverse
        reverse = {}
        for word in similars:
            # Add the word itself has emojis add them
            if word in corpusmap:
                for emoji in corpusmap[word]:
                    if not word in reverse:
                        reverse[word] = []
                    reverse[word].append((emoji, 1.0))

            # Then do similar words
            for sim, val in similars[word]:
                if sim in corpusmap:
                    for emoji in corpusmap[sim]:
                        if not word in reverse:
                            reverse[word] = []
                        reverse[word].append((emoji, val))
        print('Generated reverse')

        with open(REVERSE_NAME, 'wb') as f:
            pickle.dump(reverse, f)
        print('Saved reversed')

    with open(REVERSE_NAME, 'rb') as f:
        reverse = pickle.load(f)

    # Interactive console
    print('Enter a word to get emojis; type EXIT to stop')
    while True:
        inp = input()
        if inp == 'EXIT':
            break
        try:
            matches = reverse[inp]
            matches.sort(key=lambda x: x[1], reverse=True)
            printed = 0
            matchset = set()
            for m in matches:
                if m[0] not in matchset:
                    print('{}: {}'.format(emojis[m[0]]['char'], m[1]))
                    matchset.add(m[0])
                    printed += 1
                    if printed == 10:
                        break
        except KeyError:
            print('Sorry: I could not find any good emojis')
