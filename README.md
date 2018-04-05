# Emoji Suggester

An emoji recommending service that finds you the most similar emojis for a word with word2vec.

## Usage

First, you'll need a word2vec dataset. The [Google News dataset](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing) set works great. Make sure to un-gzip it.

Then, you can run either `bymaxdegree.py` or `bysimilarlist.py` for a console based emoji suggestion service (your console will need to support emojis), or you can run `emojiserver/emojiserver.py` to spawn a webserver.

## Methods

Things are currently done two ways (`bymaxdegree.py` and `bysimilarlist.py` contain the different implementations):

### Common implemntation

Both methods use the [emojilib](https://github.com/muan/emojilib) library to obtain a set of keywords in addition to the emoji names (also provided by this library).
From this set of emoji names and keywords, they create a set of words containing both emoji names and keywords that are later mapped back to the emoji names.

### By similarity lists

This algorithm finds the 100 most similar words for each word in the word set, then creates a map from this data. It then goes through and reversed this map, creating a new map of words to their emojis. When a word is queried, it looks up the relevant emoji in this map.

While this implementation is quite fast and memory efficient, the number of words that get mapped to emojis is quite small.

### By maximum similarity

Ideally one could use the entire word2vec model to compare the lookup word with the word set and find the closest emoji. However, the Google News dataset is quite large (3.4 GB!), so this implementation reduces the dataset to include words reasonabally similar to emojis (which is about 310 MB).

It can then for each word in the word set take its dot product with the queried word, and find the words that yield the greatest dot product.

## The suggestion server

This is a bit more organized than the other two files in the root directory, and it also has a larger feature set as more time has been spent on it.

## Slack emojis

To generate `slackemojis.json`, run the following in a developer console window on a Slack page:

```javascript
copy(JSON.stringify(Object.keys(emoji.data).filter(x=>emoji.data[x][7]==0).map(x=>({char: emoji.data[x][0], name: emoji.data[x][3], code: x}))))
```
