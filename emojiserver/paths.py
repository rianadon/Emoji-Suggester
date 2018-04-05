"""
This module defines a set of paths as well as a utility for getting local files.
"""

import os.path

filepath = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(filepath)

def file(filename):
    """Give the pathname for a file"""
    return os.path.join(parentdir, filename)

BIN_NAME = file('GoogleNews-vectors-negative300.bin')
SAVE_NAME = file('vectors.bin')
NSAVE_NAME = file('normedvectors.bin')

DP_NAME = file('dp.npy')

EMOJI_NAME = file('emojis.json')
SLACKEMOJIS_NAME = file('slackemojis.json')
PAREDEMOJIS_NAME = file('paredemojis.json')
