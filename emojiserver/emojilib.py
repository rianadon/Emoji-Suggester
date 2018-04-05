"""
This module downloads emojilib if necessary and holds its data.
It also contains functions to generate corpuses from the emoji data.

The emojilib data can be accessed by calling `emojis()`
"""

import json
import os.path
import urllib.request

from paths import EMOJI_NAME, SLACKEMOJIS_NAME, PAREDEMOJIS_NAME

EMOJI_URL = 'https://raw.githubusercontent.com/muan/emojilib/master/emojis.json'
SLACK_URL = 'https://raw.githubusercontent.com/iamcal/emoji-data/master/emoji.json'

def emojis():
    """Return the dict defined by emojib."""
    if not os.path.isfile(EMOJI_NAME):
        with open(EMOJI_NAME, 'wb') as f:
            with urllib.request.urlopen(EMOJI_URL) as response:
                f.write(response.read())
        print('Emojilib downloaded!')
    else:
        print('Emojilib already downloaded')

    with open(EMOJI_NAME, 'r', encoding='utf-8') as f:
        emojilib = json.load(f)
    print('Emojilib loaded')

    return emojilib

def pared_emojis():
    """Return the dict defined by emojislib, with Slack emoji names."""
    if not os.path.isfile(PAREDEMOJIS_NAME):
        allem = emojis()
        slackem = slackemojis()

        splitchar = lambda v: ('{:04X}'.format(ord(c)) for c in v['char'])
        charcode = lambda v: '-'.join(splitchar(v)).replace('-FE0F', '')
        allemdict = dict((charcode(v), dict(v, name=k)) for k,v in allem.items() if v['char'] is not None)
        slackemdict = dict((x['code'].upper(), x) for x in slackem)

        print(charcode(allem['weight_lifting_man']))

        emdiffs = set(allemdict.keys()).difference(slackemdict.keys())
        if len(emdiffs) != 0:
            for k in emdiffs:
                print('Warning: Slack has no emoji for {}'.format(allemdict[k]['name']))

        eminter = set(allemdict.keys()).intersection(slackemdict.keys())

        etoslack = {}

        # Start creating a mapping of emojilib names -> slack names
        for k in eminter:
            etoslack[allemdict[k]['name']] = slackemdict[k]['name'][0]

        # Find remaining mappings
        slackdiffs = set(slackemdict.keys()).difference(allemdict.keys())
        for key in slackdiffs:
            shortname = slackemdict[key]['name'][0]
            if shortname.startswith('skin-tone'):
                continue # These aren't real emojis; only modifiers
            if key == '2640' or key == '2642':
                continue # Again, gender modifiers, not emojis

            # Try removing gender modifiers
            if key.endswith('-200D-2640') or key.endswith('-200D-2642'):
                newkey = key[:-10]
                if newkey in allemdict:
                    etoslack[allemdict[newkey]['name']] = slackemdict[key]['name'][0]
                    continue

            # Now give up
            print('Warning: could not find emoji in emojilib for {}, {}'.format(shortname, key))

        for key in etoslack:
            etoslack[key] = {
                'name': etoslack[key],
                'char': allem[key]['char'],
                'keywords': allem[key]['keywords']
            }

        with open(PAREDEMOJIS_NAME, 'w', encoding='utf-8') as f:
            json.dump(etoslack, f, ensure_ascii=False)

        print('Pared emoji dict generated')

    with open(PAREDEMOJIS_NAME, 'r', encoding='utf-8') as f:
        paredemojis = json.load(f)

    return paredemojis

def slackemojis():
    """Return an array of emojis that slack supports."""
    if not os.path.isfile(SLACKEMOJIS_NAME):
        with open(SLACKEMOJIS_NAME, 'wb') as f:
            with urllib.request.urlopen(SLACK_URL) as response:
                f.write(response.read())
        print('Slack emojis downloaded!')
    else:
        print('Slack emojis already downloaded')

    with open(SLACKEMOJIS_NAME, 'r', encoding='utf-8') as f:
        slackemojiset = json.load(f)
    print('Slack emojis loaded')

    return slackemojiset

def emojicorpus(emojis):
    """Return a set of all emoji names and keywords in the given dict."""
    wordcorpus = set()
    for name in emojis:
        wordcorpus.add(name)
        for keyword in emojis[name]['keywords']:
            wordcorpus.add(keyword)
    return wordcorpus

def emojicorpusmap(emojis, vocab):
    """Return a corpus map with each word mapped to its emoji.

    The function takes in the emojilib dict and an array or dict of words to
    limit the map to.
    """
    corpusmap = {}
    for name in emojis:
        if name in vocab:
            if name not in corpusmap:
                corpusmap[name] = []
            corpusmap[name].append(name)
        for keyword in emojis[name]['keywords']:
            if keyword in vocab:
                if keyword not in corpusmap:
                    corpusmap[keyword] = []
                corpusmap[keyword].append(name)
    return corpusmap
