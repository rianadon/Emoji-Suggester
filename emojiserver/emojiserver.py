"""
This program spawns an HTTP server that serves relevant emojis based on the
request path.
"""

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np

import emojilib
import word2vec

NUM_EMOJIS = 10 # Number of emojis to print
CATEGORY_LENGTH = 8 # Disregard categories with greater or equal to this many elements

EMOJIS = emojilib.pared_emojis()
WORDCORPUS = emojilib.emojicorpus(EMOJIS)
MODEL = word2vec.normedmodel(WORDCORPUS)

CORPUSMAP = emojilib.emojicorpusmap(EMOJIS, MODEL.vocab)
WCL = list(CORPUSMAP.items())
VECTORCORPUS = word2vec.vectorcorpus(MODEL, WCL)

def similar(word, num):
    """Return the n matching emojis for a word."""
    dotprod = np.dot(VECTORCORPUS, MODEL.word_vec(word))

    # Find the matches with the most similarity
    matches = np.argpartition(dotprod, -num)[-num:]
    sortedmatches = matches[np.argsort(dotprod[matches])][::-1]

    # First find matching emojis that aren't in large categories
    goodnames = []
    goodcategories = []
    nameset = set()
    for index in sortedmatches:
        names = WCL[index][1]
        if len(names) < CATEGORY_LENGTH:
            for name in names:
                if name not in nameset:
                    goodnames.append((name, float(dotprod[index])))
                    nameset.add(name)
        else:
            goodcategories.append((names, dotprod[index]))

    # If there aren't enough then start looking at categories
    if len(goodnames) < NUM_EMOJIS:
        goodcategories.sort(key=lambda x: len(x[0]))
        for category, similarity in goodcategories:
            for name in category:
                if name not in nameset:
                    goodnames.append((name, float(similarity)))
                    nameset.add(name)
                    if len(goodnames) == num:
                        break
            if len(goodnames) == num:
                break

    return goodnames[:num]

def formatsimilar(names, emojis):
    """Return the output of `similar` but with the emojis instead of their
    names or the slack names.
    """
    ret = []
    for name, similarity in names:
        if emojis:
            ret.append((EMOJIS[name]['char'], similarity))
        else:
            ret.append((EMOJIS[name]['name'], similarity))
    return ret

class EmojiRequestHandler(BaseHTTPRequestHandler):
    """A request handler the serves up emojis for words."""

    def do_GET(self):
        """Handle GET requests."""
        req = urlparse(self.path)
        query = parse_qs(req.query, keep_blank_values=True)
        word = req.path[1:]
        try:
            num = int(query['number'][0])
        except (ValueError, KeyError, IndexError):
            num = NUM_EMOJIS

        try:
            data = formatsimilar(similar(word, num), 'emojis' in query)
        except KeyError:
            data = None

        self.send_response(200)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode('utf-8'))

    def log_message(self, format, *args):
        """Don't log anything."""
        return

if __name__ == '__main__':
    print('Starting server')
    server_address = ('', 8000)
    httpd = HTTPServer(server_address, EmojiRequestHandler)
    httpd.serve_forever()
