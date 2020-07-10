__author__ = 'TheOz'


# import reload
from importlib import reload

# import web tools
import urllib.request
import bs4
from bs4 import BeautifulSoup
import urllib.parse
from urllib.parse import urljoin

# import math
import math
exp = math.exp
sqrt = math.sqrt

# import numpy, scipy
import numpy as np
import scipy as sc

# import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# import random
import random
rand = random.random
choice = random.choice

# import time
import time
sleep = time.sleep
clock = time.clock

# import keras
#import keras
#from keras.models import Sequential
#from keras.layers import Dense, Activation

# import sklearn
from sklearn import decomposition
from sklearn import linear_model
from sklearn import cluster


# Page class
class Page(list):
    """Class for extracting information from a webpage.

    Inherits from list.
    """

    def __init__(self, u=None):
        """Initialize a Page instance with a url address.

        Arguments:
            u: string, a url address

        Attributes:
            soup: Beautiful Soup object
            text: string, the text from the webpage
            url: string, the url address
        """

        self.soup = None
        self.text = ' '
        self.url = u

    # instance methods
    def absorb(self, p):
        """Absorb the contents of one page into another.

        Arguments:
            p: Page instance

        Returns:
            None
        """

        for i in p:
            self.append(i)

        return None

    def apply(self, f):
        """Apply a transformation to each member.

        Arguments:
            f: function object, change to apply

        Returns:
            None
        """

        # transfer to copy
        c = []
        while len(self) > 0:
            c.append(self.pop())
        c.reverse()

        # apply function
        c = [f(i) for i in c]

        # redeposit
        for i in c:
            self.append(i)

        return None

    def cook(self):
        """Cook the soup by using Beautiful Soup to extract the contents.

        Arguments:
            None

        Returns:
            None
        """

        # retrieve url
        u = self.url
        try:
            o = urllib.request.urlopen(u)

        # abort if unsuccesful
        except:
            print('Could not open %s' % u)

            return None

        # get soup
        s = BeautifulSoup(o.read(), features='html.parser')
        print('got soup')
        self.soup = s

        return None

    def crop(self, a, b):
        """Crop list of strings to those between two bracketting conditions.

        Arguments:
            a: function object, criteria for beginning the bracket
            b: function object, criteria for ending the bracket

        Returns:
            None
        """

        # empty self into a copy
        c = []
        while len(self) > 0:
            c.append(self.pop())
        c.reverse()

        # go through each member
        k = False
        for i in c:

            # ending condition
            if b in i:
                break

            # deposit if k is True
            if k:
                self.append(i)

            # starting condition
            if a in i:
                k = True

        return None

    def deposit(self):
        """Deposit all lines of text into the page.

        Arguments:
            None

        Returns:
            None
        """

        # get text
        t = self.text

        # deposit lines
        s = t.splitlines()
        for i in s:
            self.append(i)

        return None

    def dig(self,n):
        """Dig for text a certain number of layers deep to populate the page.

        Arguments:
            n: integer, number of layers

        Returns:
            None
        """

        # get contents
        c = self.soup.contents

        # loop n times
        for i in range(n):
            c = [i.contents for i in c if i.string is None]
            c = [i for j in c for i in j]

        # get text
        t = [i.string for i in c if i.string is not None]

        # populate page
        for i in t:
            self.append(i)

        return None

    def dice(self, d=' '):
        """Dice all strings into smaller strings based on a particular character.

        Arguments:
            d=' ': string, character to use for splitting strings, space by default

        Returns:
            None
        """

        # empty into copy
        c = []
        while len(self) > 0:
            c.append(self.pop())
        c.reverse()

        # split
        c = [i.split(d) for i in c]

        # reform list
        c = [i for j in c for i in j]

        # redeposit
        for i in c:
            self.append(i)

        return None

    def keep(self, f):
        """Keep all words according to some criteria.

        Arguments:
            f: function object, word is kept if f evaluates to True

        Returns:
            None
        """

        # empty into copy
        c = []
        while len(self) > 0:
            c.append(self.pop())
        c.reverse()

        # apply condition
        c = [i for i in c if f(i)]

        # redeposit
        for i in c:
            self.append(i)

        return None

    def read(self, n):
        """Read in data from a file.

        Argument:
            n: string, file name

        Return:
            None
        """

        # open file
        f = open(n, 'r')

        # readin lines
        w = []
        for i in f:
            self.append(i[:-1])

        # close flie
        f.close()

        return None

    def skim(self):
        """Skim off one of each member in the case where members are duplicated.

        Arguments:
            None

        Return:
            None
        """

        # pop into a copy
        l = len(self)
        c = [self.pop() for n in range(l)]

        # remove duplicates
        for n in range(l):

            # get next element and append if origianl
            e = c.pop()
            if e not in self:
                self.append(e)

        return None

    def transcribe(self):
        """Transcribe the text portion from the Beautiful Soup object.

        Arguments:
            None

        Returns:
            None
        """

        # search in string attribute
        s = self.soup
        g = s.string

        # but look in contents if it is missing
        if g is None:
            c = s.contents

            # result text
            t = ''

            # treat each node like a new soup
            for n in c:
                p = Page()
                p.soup = n
                p.transcribe()
                t += p.text + '\n'

            # store in attribute
            self.text = t

            return None

        # otherwise strip
        else:
            t = g.strip()
            self.text = t

            return None

    def write(self, n):
        """Write the data to file for later retrieval.

        Argument:
            n: string, file name

        Returns:
            None
        """

        # open file
        f = open(n, 'w')

        # write each word with a space
        for i in self:
            f.write(i + '\n')

        # close file
        f.close()

        return None


# Song class
class Song(Page):
    """Class for containing the list of lyrics from a song.

    Inherits from Page."""

    def __init__(self, f, n, u):
        """Initialize a song as an empty list.

        Arguments:
            f = string, file
            n = string, name
            u = string, url address

        Attributes:
            artist: string, name of song's artist
            name: string, name of song
            selected: boolean, selected for study?
            soup: Beautiful Soup object
            url: string, url address
        """

        # attributes
        self.file = f
        self.name = n
        self.selected = False
        self.soup = None
        self.url = u

    # static methods
    @staticmethod
    def _filter(l):
        """Filter non unicodeable characters from lines.

        Arguments:
            l: string, lyric line

        Returns:
            string, filtered line
        """

        # dictionary of replacements
        d = {u'\u2019': "'", u'\xa0': '', u'\u201c': '"', u'\u201d': '"', u'\u2026': '...', u'\u2014': '--'}
        d[u'\u2018'] = '-'
        d[u'\u2012'] = '-'

        # begin filter
        f = ""

        # go through each character
        for i in l:

            # attempt to add
            try:
                f += str(i)

            # otherwise implant replacement
            except:
                if i in d.keys():
                    print('replaced an ascii: ', i)
                    f += d[i]
                else:
                    print(l)
                    print(i)
                    str(i)

        return f

    # instance methods
    def __repr__(self):
        """Create a string for a song object.

        Arguments:
            None

        Returns:
            string
        """

        # create string
        s = '< Song object: ' + str(self.name) + ' >'

        return s

    def copy(self):
        """Copy the song's information into a new instance.

        Arguments:
            None

        Returns:
            Song instance
        """

        # make copy
        c = Song(self.file, self.name, self.url)

        # copy lines:
        for i in self:
            c.append(i)

        return c

    def jot(self):
        """Add the song to the artist's file.

        Arguments:
            None

        Returns:
            None
        """

        # open file for appending
        f = open(self.file, 'a')

        # create lines
        l = [i + '\n' for i in self]

        # add headers and ender
        l = [self.url + '\n'] + l
        l = [self.name + '\n'] + l
        l.append('[end]\n')

        # filter out non unicode
        l = [Song._filter(i) for i in l]

        # write to file
        f.writelines(l)

        # close file
        f.close()

        return None

    def transcribe(self):
        """Transcribe the lyrics of the song.

        Arguments:
            None

        Returns:
            None
        """
        # cook soup
        self.cook()
        p = self.soup

        # extract lyrics
        c = p.contents[2]
        d = c('div')
        d = [i for i in d if 'class' not in [j[0] for j in i.attrs]]

        # find lyrics
        y = d[1]
        y = [i.string for i in y if i.string is not None]

        # eliminate disclaimer
        y = y[2:]

        # eliminate newlines
        y = [i.strip('\n') for i in y]

        # populate
        for i in y:
            self.append(i)

    def view(self):
        """print the contents to screen.

        Arguments:
            None

        Returns:
            None
        """

        # print to screen
        for i in self:
            print(i)

        return None


# Artist class
class Artist(Page):
    """Class for extracting an artist's lyrics from the azlyrics website.

    Inherits from Page.
    """

    az_lyrics = 'http://www.azlyrics.com/'

    def __init__(self, name):
        """Initialize an Artist instance with an artist's name.

        Arguments:
            name: string, an artist name

        Attributes:
            file: string, name of associated file
            lyrics: list of strings, lyrics from file
            name: string, an artist's name
            songs: dictionary, maps song name to link
            soup: Beautiful Soup object
            url: string, the url address
        """

        # collapse name
        name = name.lower().replace(' ','')

        self.file = 'Artists/' + name + '.txt'
        self.lyrics = []
        self.name = name
        self.songs = {}
        self.soup = None
        self.url = Artist.az_lyrics + name[0] + '/' + name + '.html'

    def __repr__(self):
        """Create a string for viewing a Band instance.

        Arguments:
            None

        Returns:
            string
        """

        # create string
        s = '< Artist object: ' + self.name + ' >'

        return s

    # static methods
    @staticmethod
    def gauge(trial, target, window=3):
        """Gauge the closeness of a trial string to a target string by counting common letter groupings

        Arguments:
            trial: string
            target: string
            window=3: integer, length of letter groupings to examine

        Returns:
            integer, relatedness score
        """

        # break target into groupings
        partials = []
        partial = target
        for round in range(window):
            partials.append(partial)
            partial = partial[1:]

        # remerge as groupings
        groupings = zip(*partials)
        groupings = [''.join(group) for group in groupings]

        # count occurrences in trial
        occurrences = [group for group in groupings if group in trial]
        score = len(occurrences)

        return score

    # instance methods
    def catalog(self):
        """Catalog the songs by the band.

        Arguments:
            None

        Returns:
            None
        """

        # attempt to fetch from file
        try:
            self.fetch()

        # otherwise attempt to find the url
        except IOError:
            try:
                self.snoop()
                self.jot()

            # otherwise print error
            except TypeError:
                print('Artist %s was not found' % self.name)

        return None

    def fetch(self):
        """Fetch the song list from a file.

        Arguments:
            None

        Returns:
            None
        """

        # try to open file
        try:
            f = open(self.file, 'r')
        except:
            raise IOError

        # readlines into list
        l = [i.strip('\n') for i in f]

        # close file
        f.close()

        # populate songs
        l.reverse()
        p = l.pop()
        while p != '[end]':
            q = l.pop()
            self.songs[p] = q
            p = l.pop()

        # add rest to lyrics
        while len(l) > 0:
            p = l.pop()
            self.lyrics.append(p)

        return None

    def gather(self):
        """Gather already saved lyrics into song objects.

        Arguments:
            None

        Returns:
            None
        """

        # get lyrics
        y = self.lyrics

        # make lists
        l = []
        g = []
        for i in y:

            # append to growing song unless end is reached
            if i != '[end]':
                g.append(i)

            # otherwise start new list
            else:
                l.append(g)
                g = []

        # create song objects
        for i in l:

            # new song
            s = Song(self.file, i[0], i[1])

            # fill in lyrics
            for j in i[2:]:
                s.append(j)

            # append song
            self.append(s)

        return None

    def grab(self, n=None):
        """Grab a song from the website.

        Arguments:
            n=None: string, name of song to search for

        Returns:
            None
        """

        # find song to grab
        if n:
            n = n.lower().replace(' ','')

        # list of songs already in the repertoire
        r = [i.name for i in self]

        # list of songs in catalog but not in repertoire
        y = self.songs.keys()
        y = [i for i in y if i not in r]

        # pick first name in catalog
        if n is None:
            n = y[0]

        # or closest to requested name
        else:
            q = [(i, self.gauge(n, i)) for i in y]
            q.sort(key=lambda x: x[1], reverse=True)
            n = q[0][0]

        # print song name
        print(n)

        # make a song object
        s = Song(self.file, n, self.songs[n])

        # create random pause up to 5 seconds
        sleep(rand() * 5)

        # retrieve lyrics, add to repertoire
        s.transcribe()
        self.append(s)

        # write to file
        s.jot()

        return None

    def jot(self):
        """Jot down the list of songs and links into a file.

        Arguments:
            None

        Returns:
            None
        """

        # open file
        f = open(self.file,'w')

        # construct lines
        l = []
        for k, i in self.songs.items():
            l.append(k + '\n')
            l.append(i + '\n')

        # add ending tag
        l.append('[end]\n')

        # writelines
        f.writelines(l)
        f.close()

    def seek(self, letter=None, entries=3):
        """Seek the artist spelling in the az lyrics pages.

        Arguments:
            letter: string, the letter in the index to search
            entries=3: integer, number of closest matches given

        Returns:
            None
        """

        # take beginning letter of name by default
        if letter is None:
            letter = self.name[0]

        # make url
        url = self.az_lyrics + letter + '.html'

        # make a page instance and cook
        names = Page(url)
        names.cook()

        # retrieve links from soup
        links = names.soup('a')
        links = [link.get('href', None) for link in links]
        links = [link for link in links if link is not None]

        # prune to just ids, the last word in the url
        ids = [link.split('/')[-1] for link in links]
        ids = [id.split('.')[0] for id in ids]

        # gauge the closeness of each id to the name in question
        scores = [(id, self.gauge(id, self.name)) for id in ids]
        scores.sort(key=lambda score: score[1], reverse=True)

        # print best matches
        print(' ')
        for match in scores[:entries]:
            print(match[0])
        print(' ')

        return None

    def snoop(self):
        """Snoop on the website, retrieving a list of songs and links to those songs.

        Arguments:
            None

        Returns:
            None
        """

        # grab the web info
        self.cook()
        soup = self.soup

        # try to retrieve links
        links = soup('a')

        # prune to song title links
        links = [link.get('href', None) for link in links]
        links = [link for link in links if link is not None]
        links = [link for link in links if link.startswith('../lyrics')]
        links = [link.split('../')[1] for link in links]

        # prune to album titles
        titles = [link.split('/')[-1] for link in links]
        titles = [title.split('html')[0] for title in titles]
        titles = [title.strip('.') for title in titles]

        # add opening url
        reference = Artist.az_lyrics
        links = [reference + link for link in links]

        # make dictionary
        songs = {}
        for title, link in zip(titles, links):
            songs[title] = link

        # put into attributes
        self.songs = songs

        return None

    def view(self):
        """View the discography of the band.

        Arguments:
            None

        Returns:
            None
        """

        # view
        for index, song in enumerate(self):

            # determine if song is selected
            indicator = '   '
            if song.selected:
                indicator = ' X '

            # print to screen
            print(indicator + str(index) + ': ' + str(song))

        return None


# Node class
class Node(list):
    """Class for a hidden node in an artificial neural net.

    Inherits from list.
    """

    def __init__(self, w):
        """Initialize a node as a list of three dictionaries of weights.

        Arguments:
            w: list of strings, words

        Attributes:
            activity: float, activity of the node
            slopes: list of dictionaries mapping words to slopes with respect to error, dE/dw
        """

        # set activity to 0.0
        self.activity = 0.0

        # initialize slopes
        self.slopes = [{}, {}, {}]

        # beginning and ending symbols
        a = ('<', None, '>')

        # for each set of weights
        for n in range(3):

            # append dictionary
            self.append({})

            # add beginning or ending symbol
            u = w[:]
            if a[n]:
                u.append(a[n])

            # for each word
            for i in u:

                # set the slope to zero
                self.slopes[n][i] = 0.0

                # set weight between -4.0 and 0.0, biased toward -4.0
                self[n][i] = (rand() ** 24) * 4.0 - 4.0

    def __repr__(self):
        """print out node object signifier instead of actual list of weights.

        Arguments:
            None

        Returns
            string
        """

        return '< Node object >'

    def compress(self):
        """Compress the range of weights to between -2 and 2.

        Arguments:
            None

        Returns:
            None
        """

        # for all groups
        for i in self:

            # find max and minimum
            v = i.values()
            a = max(v)
            b = min(v)

            # compare to 2 and -2
            c = abs(2.0 / a)
            d = abs(2.0 / b)

            # find maximum
            m = min([c, d])

            # multiply all weights
            for k in i.keys():
                i[k] *= m

        return None

    def erase(self):
        """Erase weights of neuron, restoring all weights to 0.0.

        Arguments:
            None

        Returns:
            None
        """

        # for each set
        for n in range(3):

            # erase all weights
            for k in self[n].keys():
                self[n][k] = 0.0

            # and slopes
            for k in self.slopes[n].keys():
                self.slopes[n][k] = 0.0

        # zero out activity
        self.activity = 0.0

        return None

    def view(self, n=7):
        """View the weights of the neurons in order of highest first.

        Arguments:
            n=7: number of highest weighted listings

        Returns:
            None
        """

        # for each set
        for s in range(3):

            # print header
            print('weights %d:' % s)

            # get all weights
            w = self[s].items()

            # sort
            w.sort(key=lambda x: x[1], reverse=True)

            # print first few
            for i, j in w[:n]:
                print(i, j)

            # spacer
            print(' ')

        return None

    def weight(self, n, w, t):
        """Set the weight of a connection to the neuron.

        Arguments:
            n: integer, group of connections to set
            w: string, word to set
            t: float, weight to set

        Returns:
            None
        """

        # set weight
        self[n][w] = t

        return None


# Neural Net class
class NeuralNet(list):
    """Class NeuralNet defines a set of nodes and training operations upon them.

    Inherits from list.
    """

    def __init__ (self, n, x):
        """Initialize a neural net with a number of nodes, inputs, and outputs.

        Arguments:
            n: integer, number of nodes
            x: Lexicon instance

        Attributes:
            lexicon: Lexicon instance
            predictions: dictionary mapping possible third words to probabilities
            reconstructions: dictionary mapping possible first words to probabilities
        """

        # set lexicon
        self.lexicon = x

        # get words
        w = list(x.words)

        # add nodes
        for i in range(n):
            self.append(Node(w))

        # initiate predictions
        self.predictions = {i: 0.0 for i in w + ['>']}

        # initiate reconstructions
        self.reconstructions = {i: 0.0 for i in w + ['<']}

    def appraise(self, t=None, p=False):
        """Calculate the combined error of all given triples.

        Arguments:
            t=None: list of strings, triples, all by default
            p=False: boolean, print all errors?

        Returns:
            float, combined error
        """

        # set triples
        if t is None:
            t = self.lexicon

        # go through each triple
        l = []
        for i in t:
            e = self.examine(i)
            f = self.examine(i, False)
            l.append((i, e, f))

        # print to screen
        if p:

            # sort for highest error last
            l.sort(key=lambda x: x[1] + x[2])

            # print list
            for i in l:
                print(i)

        # sum all errors, forwards and backwards
        a = [i[1] for i in l]
        b = [i[2] for i in l]
        c = sum(a) + sum(b)

        return c

    def compress(self):
        """Compress all neurons in the network.

        Arguments:
            None

        Returns:
            None
        """

        # compress all neurons in the network
        for i in self:
            i.compress()

        return None

    def diagnose(self, u, d, n, w, f=True):
        """Calculate the slope of error for a weight in the network.

         Arguments:
            u: string, two inputs and one output
            d: integer, node number
            n: integer, weight set number
            w: string, word key
            f=True: boolean, forward direction?

        Returns:
            tuple:
                float: error
                float: numerical slope,
                float: analytical slope
        """

        # empty stored slopes
        self.empty()

        # get initial error
        a = self.examine(u, f)

        # get slope info
        self.reflect(u, f)
        r = self[d].slopes[n][w]

        # clear slopes
        self.empty()

        # adjust weight
        self[d][n][w] += 0.001

        # get second error
        b = self.examine(u, f)

        # readjust weights
        self[d][n][w] -= 0.001

        # calculate slope
        s = (b - a) / 0.001

        return a, s, r

    def empty(self):
        """Empty all stored slopes.

        Arguments:
            None

        Returns:
            None
        """

        # for each node
        for i in self:

            # and each set of weights
            for n in range(3):

                # and each weight
                for j in i.slopes[n].keys():

                    # set slope to zero
                    i.slopes[n][j] = 0.0

        return None

    def examine(self, u, f=True):
        """Calculate the error based on stored predictions.

        Arguments:
            u: string, triplet
            f=True: boolean, examine forward?

        Returns:
            float, the prediction error
        """

        # forward direction
        if f:
            u = u.split()
            o = u.pop()
            u = ' '.join(u)
            p = self.predictions

        # or backwards
        else:
            u = u.split()
            o = u.pop(0)
            u = ' '.join(u)
            p = self.reconstructions

        # begin error
        e = 0.0

        # set lexicon pointer
        x = self.lexicon

        # go through all predictions
        self.predict(u, f)
        for k, v in p.items():

            # set target
            t = x.similarities[o][k]

            # error = 1/2 (t - p)^2
            a = ((t - v) ** 2) / 2

            # add to error
            e += a

        return e

    def pick(self, f=True, c=0.1):
        """Pick an answer from a weighted list of predictions.

        Arguments:
            f=True: boolean, forward direction?
            c=0.1: float, cutoff value for considered predictions

        Returns:
            string, predicted word
        """

        # get predictions
        if f:
            p = self.predictions
        else:
            p = self.reconstructions

        # find top value
        t = max(p.values())

        # pare down predictions by cutoff value
        p = {i: j for i, j in p.items() if (j / t) > c}

        # get random entry
        r = rand() * sum(p.values())

        # accumulate until sum is greater than r
        a = 0.0
        w = ' '
        for i, j in p.items():

            # add to accumulations
            a += j

            # stop if greater than random
            if a >= r:
                w = i
                break

        return w

    def predict(self, u, f=True, n=None):
        """Predict the activities of outputs based on given inputs.

        Arguments:
            u: string of words, generally two
            f=True: boolean, predict forward?
            n=None: integer, number of top predictions to print

        Returns:
            None
        """

        # activity function (logistic)j.r
        ac = lambda x: 1.0 / (1.0 + exp(-x))

        # split words and keep first two
        u = u.split()[:2]

        # determine indices based on prediction direction
        if f:
            a, b, c = (0, 1, 2)
            v, w = u
            p = self.predictions
        else:
            a, b, c = (2, 1, 0)
            w, v = u
            p = self.reconstructions

        # calculate activity for each node
        for i in self:

            # begin input total
            t = 0.0

            # add weight of first word
            t += i[a].get(v, 0.0)

            # add weight of middle word
            t += i[b].get(w, 0.0)

            # convert to activity
            i.activity = ac(t)

        # for each prediction
        for i in p.keys():

            # begin total
            t = 0.0

            # sum weight from each node
            for j in self:

                # get activity and weight
                t += j.activity * j[c].get(i, 0.0)

            # convert to activity
            p[i] = ac(t)

        # if optional number of predictions is given
        if n:

            # sort predictions
            y = p.items()
            y.sort(key=lambda x: x[1], reverse=True)
            y = y[:n]

            # print to screen
            for i in y:
                print(i)
            print(' ')

        return None

    def reflect(self, u, f=True):
        """Reflect on the predictions, determining slopes of the error function with respect to each weight.

        Arguments:
            u: string, generally of two input words and one output word
            f=True: boolean, reflect on forward predictions?

        Returns:
            None
        """

        # access lexicon
        x = self.lexicon

        # split the input string, retaining only three
        u = u.split()[:3]

        # determine forward indices
        if f:
            v, w, o = u
            a, b, c = (0, 1, 2)
            p = self.predictions

        # or backward indices
        else:
            o, w, v = u
            a, b, c = (2, 1, 0)
            p = self.reconstructions

        # determine target likelihoods based on functional similarity
        t = {i: x.similarities[i][o] for i in p.keys()}

        # total input to node j is the sum of all inputs multiplied by their weights
        # xj = i{ ui * wij

        # activity of node j is the logistic function of total weighted inputs
        # aj = f(xj)

        # total input to word k is the sum of all activities multiplied by their weights
        # zk = j{ aj * yjk

        # total prediction of word k is the logistic function of weighted inputs
        # pk = f(zk)

        # the error for word k is 1/2 the squared difference between prediction and target
        # ek = 1/2 (pk - tk)^2

        # the total error is the sum of error for all words
        # e = k{ ek

        # logistic function is
        # f = 1 / (1 + e^-x)
        # df/dx = f * (1 - f)

        # slope of error with respect to input weights is
        # de/dwij = k{ dek/dwij
        # = k{ dek/dpk * dpk/dwij
        # = k{ dek/dpk * dpk/dzk * dzk/dwij
        # = k{ dek/dpk * dpk/dzk * dzk/daj * daj/dwij
        # = k{ dek/dpk * dpk/dzk * dzk/daj * daj/dxj * dxj/dwij

        # de/dwij = k{ (pk - tk) * pk * (1 - pk) * yjk * aj * (1 - aj) * ui

        # determine slopes for first and second input weights
        for i in self:

            # activity
            y = i.activity

            # for each predicted word
            for k, j in p.items():

                # calculate slope of error
                s = (j - t[k]) * j * (1.0 - j) * i[c][k] * y * (1.0 - y) * 1.0

                # add to slopes
                i.slopes[a][v] += s
                i.slopes[b][w] += s

        # slope of error with respect to output weights is
        # de/dyjk = dek/dyjk
        # = dek/dpk * dpk/dyjk
        # = dek/dpk * dpk/dzk * dzk/dyjk

        # de/dyjk = (pk - tk) * pk * (1 - pk) * aj

        # determine slopes for last weights
        for i in self:

            # activity
            y = i.activity

            # for each output
            for k, j in p.items():

                # calculate slope of error
                s = (j - t[k]) * j * (1.0 - j) * y

                # add to slopes
                i.slopes[c][k] += s

        return None

    def revise(self, l=0.1):
        """Update the weights in the neural net based on learning rate and accumulated slopes.

        Arguments:
            l=0.1: float, learning rate

        Returns:
            None
        """

        # for each node
        for i in self:

            # and for each group
            for j in range(3):

                # and for each key
                for k in i[j].keys():

                    # update the weight and clear the slope
                    i[j][k] += -l * i.slopes[j][k]
                    i.slopes[j][k] = 0.0

        # compress
        self.compress()

        return None

    def view(self, h=7):
        """View the highest weights of each neuron.

        Arguments:
            h=7: number of highest weights to display

        Returns:
            None
        """

        # print( each node
        for n, i in enumerate(self):

            # print header
            print('Node %d:' % n)

            # view node
            i.view(h)

        return None


# Word class
class Word(dict):
    """Class for containing a word's contextual information.

    Inherits from dict, where each key is a second word and each value is a list of integers.  A positive integer indicates
    an instance in which the second word was found a number of spaces after the first word.  Negative integers represent spaces previous.
    """

    def __init__(self, word):
        """Initialize a word instance with a string for the word.

        Arguments:
            word: string, the word itself

        Attributes:
            frequency: float, relative frequency of the word in the lexicon.
            name: string, the word itself
        """

        # set a key for the word itself to an empty list
        self[word] = []

        # set attributes
        self.frequency = 0.0
        self.name = word

    # instance methods
    def note(self, word, position):
        """Take note that a secondary word has occurred at a certain position in relationship to this word.

        Arguments:
            word: string, secondary word
            position: integer, used as a key for the set of secondary words at this position

        Returns:
            None
        """

        # initialize an entry if none found
        if word not in self:
            self[word] = []

        # append an instance to the entry
        self[word].append(position)

    def view(self):
        """View the context of the word.

        Arguments:
            None

        Returns:
            None
        """

        # print word
        print(self.name + ':')
        print(' ')

        # sort entries into dictionary by position
        context = {}

        # loop through each key
        for word, positions in self.items():

            # and each position
            for position in positions:

                # create empty set if needed
                if position not in context.keys():
                    context[position] = set()

                # add entry
                context[position].add(word)

        contexts = context.keys()
        contexts.sort()

        # print each set
        for position in contexts:
            print(str(position) + ': ' + ', '.join(list(context[position])))

        # spacer
        print(' ')

        return None


# Lexicon class
class Lexicon(dict):
    """Class for containing the linguistic deconstruction of the bard's lyrics.

    Inherits from dict.  Each key is a word, each value is a dictionary of its attributes.
    """

    def __init__(self, transcript):
        """Initialize a lexicon instance with a list of lyric lines.

        Arguments:
            transcript: list of lists of strings, the lyrics from each song

        Attributes:
            bins: dictionary, mapping frequency bin number to words in that frequency group
            frequencies: dictionary, mapping words to their frequencies of occurrence
            histogram: dictionary, mapping lyric length to tallies of lines with that length
            lyrics: list of all lyrics
            mapping: dictionary mapping each word to a list of lists of indices, the positions in each song
            mirror: dict mapping words to their position along the alphabetical list
            sequences: list of lists of strings, the words in their song orders
            vocabulary: alphabetical list of the words
        """

        # remove one word lyrics or single word repeating lyrics
        transcript = [[lyric for lyric in song if len(set(lyric.split())) > 1] for song in transcript]

        # remove all lyrics that have any word more than once
        transcript = [[lyric for lyric in song if self.inspect(lyric) == False] for song in transcript]

        # combine all songs into a list of lyrics and remove duplicates
        lyrics = [lyric for song in transcript for lyric in song]
        lyrics = list(set(lyrics))
        lyrics.sort()
        self.lyrics = lyrics

        # initalize vocabulary attribute by taking all words from the lyrics and removing duplicates
        vocabulary = [word for line in lyrics for word in line.split()]
        vocabulary = list(set(vocabulary))
        vocabulary.sort()
        self.vocabulary = vocabulary

        # create mirror attribute by inverting the words and position labels
        mirror = {word: label for label, word in enumerate(vocabulary)}
        self.mirror = mirror

        # initialize entries for all words
        for word in vocabulary:
            self[word] = {}

        # initialize histogram of line lengths
        self.histogram = {}

        # initialize frequencies dictionary and bins
        self.frequencies = {word: 0 for word in vocabulary}
        self.bins = None

        # break each song into just words
        sequences = [[word for lyric in song for word in lyric.split()] for song in transcript]
        self.sequences = sequences

        # form mapping
        number = len(sequences)
        mapping = {word: [[] for round in range(number)] for word in vocabulary}
        for index, group in enumerate(sequences):

            # word by word...
            for position, member in enumerate(group):
                mapping[member][index].append(position)

        # populate
        self.mapping = mapping

    def __repr__(self):
        """print the alhphabetized list of words as the representative object printed to the screen.

        Arguments:
            None

        Returns:
            string, alphabetized string representation of the list of the lexicon's keys
        """

        return str(self.vocabulary)

    # instance methods
    def absorb(self, contexts):
        """Absorb the deconstructed lyric into the grand contextual library in the lexicon.

        Arguments:
            contexts: list of (string, dict) tuples, the words and their context dictionaries

        Returns:
            None
        """

        # absorb each word
        for word, record in contexts:

            # and each contextual record
            for context, distance in record.items():

                # make an entry or overwrite previous entry
                self[word][context] = distance

        return None

    def census(self):
        """Conduct a census of line lengths amongst the lyrics.

        Arguments:
            None

        Returns:
            None
        """

        # record the line length of each lyric
        record = {}
        for lyric in self.lyrics:

            # measure and update tally, creating entry if necessary
            length = len(lyric.split())
            record[length] = record.setdefault(length, 0) + 1

        # create histogram
        total = sum(record.values())
        record = [(key, value) for key, value in record.items()]
        record.sort(key=lambda x: x[0])
        histogram = [(0, 0)]
        for length, tally in record:

            # add to the running total
            tallies = tally + histogram[-1][1]
            histogram.append((length, tallies))

        # divide by total for frequencies
        histogram = [(length, float(tallies) / total) for length, tallies in histogram]

        # populate attribute
        self.histogram = histogram

        return None

    def decode(self, features):
        """Decode a feature vector in numerical labels into the corresponding word labels.

        Arguments:
            features: list of tuples, feature label and magnitude

        Returns:
            list of tuples, feature identity and magnitude
        """

        # translate from labels to identities
        identities = [(self.vocabulary[label], magnitude) for label, magnitude in features]

        return identities

    @staticmethod
    def deconstruct(arrangement):
        """Deconstruct an arrangement of words into a dictionary of context positions, recording the position of each
        word compared to all others.

        Arguments:
            arrangement: list of (number, string) tuples, the position and name of each word in the arrangement

        Returns:
            list of (string, dict) tuples, the words and context dictionaries of all words in the arrangement
        """

        # process enumerate objects if necessary
        arrangement = [(position, word) for position, word in arrangement]

        # compare each word in the arrangement
        contexts = []
        for position, word in arrangement:

            # to every other word in the arrangement
            record = (word, {})
            for positionii, wordii in arrangement:

                # exclude entries from the same word
                if wordii != word:

                    # calculate the distance and add to dictionary
                    distance = positionii - position
                    record[1][wordii] = distance

            # add to list of contexts
            contexts.append(record)

        return contexts

    def digest(self, resolution=3):
        """Digest all the lyrics, deconstructing the contexts of each word and adding them to the lexicon.

        Arguments:
            resolution=5: integer, number of frequency bins

        Returns:
            None
        """

        # digest each lyric
        frequencies = self.frequencies
        total = 0
        history = []
        for lyric in self.lyrics:

            # deconstruct and absorb the contexts
            contexts = self.deconstruct(enumerate(lyric.split()))
            self.absorb(contexts)

            # compare to history, filtering out lyrics with more than 70% similar words to lyrics already counted
            words = lyric.split()
            words = set(words)
            comparisons = [float(len(words & record)) / len(words | record) > 0.7 for record in history]

            # if the lyric is sufficiently different
            if True not in comparisons:

                # add the counts to the frequencies
                for word in words:
                    frequencies[word] += 1

                # increment the total number of lines
                total += 1

                # and add to the history
                history.append(words)

        # divide by total number of lyrics
        frequencies = {word: float(count) / total for word, count in frequencies.items()}
        self.frequencies = frequencies

        # sort by frequency
        frequencies = [(key, value) for key, value in frequencies.items()]
        frequencies.sort(key=lambda x: x[1])

        # sort into frequency bins
        bins = {label: [] for label in range(resolution)}
        for label in range(resolution - 1):

            # giving the first bin one word, the second two, the third four
            members = 2 ** label
            for count in range(members):
                bins[label].append(frequencies.pop()[0])

        # last bin gets the remainder
        bins[resolution -1] = [word for word, frequency in frequencies]

        # bins attribute
        self.bins = bins

        return None

    def encode(self, identities):
        """encode a set of words and average contexts into the appropriate numerical feature labels.

        Arguments:
            features: list of tuples, feature label and magnitude

        Returns:
            list of tuples, feature identity and magnitude
        """

        # translate from labels to identities
        features = [(self.mirror[word], magnitude) for word, magnitude in identities]

        return features

    @staticmethod
    def inspect(lyric):
        """Inspect a lyric for a double occurrence of any word.

        Arguments:
            lyric: string, a song lyric

        Returns:
            boolean, duplicate word present?
        """

        # add word counts to a dictionary
        tallies = {}
        for word in lyric.split():
            tallies[word] = tallies.setdefault(word, 0) + 1

        # check for tallies greater than 1
        duplicates = False
        test = [tally > 1 for tally in tallies.values()]
        if True in test:
            duplicates = True

        return duplicates

    def locate(self, query):
        """View the contexts of a word.

        Arguments:
            query: string, word in question

        Returns:
            None
        """

        # print word
        print(query + ':')
        print(' ')

        # sort entries into dictionary by position, adding self for 0 position
        contexts = {0: [query]}

        # loop through each key
        for word, position in self[query].items():

            # create empty list if needed
            if position not in contexts.keys():
                contexts[position] = []

            # add entry
            contexts[position].append(word)

        positions = contexts.keys()
        positions.sort()

        # print each list
        for position in positions:
            print(str(position) + ': ' + ', '.join(contexts[position]))

        # spacer
        print(' ')

        return None

    def reduce(self, lyric):
        """Reduce a line to single set of numerically labeled features representing the contents.

        Arguments:
            lyric: string, a line of words

        Returns:
            list of tuples, label and magnitude for each feature
        """

        # tally each word
        tallies = {}
        for word in lyric.split():

            # make an entry if there is not yet one
            if word not in tallies.keys():
                tallies[word] = 0

            # add tally
            tallies[word] += 1

        # convert to feature labels
        features = [(self.mirror[word], magnitude) for word, magnitude in tallies.items()]
        features.sort(key=lambda x: x[0])

        return features

    def trim(self):
        """Trim off any zero entries in the context dictionaries for each word.

        Arguments:
            None

        Returns:
            None
        """

        # for each word in the lexicon
        for word in self.keys():

            # retrieve items
            contexts = self[word].items()

            # remove zeroes
            contexts = [(context, distance) for context, distance in contexts if distance != 0]

            # reinstate
            self[word] = dict(contexts)

        return None


# Sample class
class Sample(dict):
    """Class for containing the information at a datapoint

    Inherits from dict where each key is an attribute of the datapoint.
    """

    def __init__(self, name, features):
        """Initialize a sample instance with its name and dictionary of vector components.

        Arguments:
            name: string, the unique identifier of the particular sample
            features: list of tuples, feature and feature quantity
        """

        # initialize the sample with name and vector as its first keywords
        self['name'] = name
        self['features'] = features

        # Give the sample a cluster label of 0 by default
        self['cluster'] = 0

    def __repr__(self):
        """Create a string to print to the screen

        Arguments:
            None

        Returns:
            Sample string
        """

        # make string
        sample_string = '< Sample object: %s >' % (self['name'])

        return sample_string

    def view(self):
        """print the sample to the screen along with its attributes:

        Arguments:
            None

        Returns:
            None
        """

        # retrieve attributes
        cluster = self['cluster']
        name = self['name']
        features = self['features']

        # retrieve list of all other keys
        main_keys = ('name', 'cluster', 'features')
        leftovers = {key: value for key, value in self.items() if key not in main_keys}

        # construct string
        sample_string = '(%d) %s: ' % (cluster, name)
        for key, value in leftovers.items():

            # append other attributes
            sample_string += '%s: %s ' % (key, str(value))

        # add vector last
        sample_string += 'features: ' + str(features)

        # print to screen
        print(sample_string)

        return None


# Grammar class
class Grammar(list):
    """Class for modeling the linguistic structure in the lexicon.

    Inherits from list where each entry is a Sample instance.
    """

    def __init__(self, data, biases=None):
        """Initialize a grammar instance with a dataset feature vectors.

        Arguments:
            biases=None: dictionary mapping feature label to feature bias
            data: list of tuples, a sample label and a list of feature tuples
            dimensions: number of dimensions in the feature space

        Attributes:
            axes: list of lists of floats, the pca components
            biases: dictionary mapping feature label to feature bias
            centers: list of arrays, the cluster centers
            clusters: list of lists of strings, the members in each cluster
            dimensions: integer, number of dimensions
            kmeans: kmeans sklearn object
            labels: labels for all features
            offset: list of floats, the mean point in the pca projection
            mirror: dictionary mapping feature label to feature id
            pca: pca sklearn objeact
            projection: the projection of the data in the pca space
            registry: dictionary mapping sample names to their sample numbers
            regression: linear regression sklearn object
            reconstructions: list of tuples, names of reconstructed points and their projections
        """

        # generate and deposit sample instances from data
        for name, features in data:

            # create sample instance
            sample = Sample(name, features)
            self.append(sample)

        # generate registry
        registry = {sample['name']: position for position, sample in enumerate(self)}
        self.registry = registry

        # make list of all feature labels
        labels = [feature[0] for sample in self for feature in sample['features']]
        labels = list(set(labels))
        labels.sort()
        self.labels = labels

        # default biases to 1
        if biases is None:
            biases = {label: 1.0 for label in labels}

        # create ordered list for biases
        biases = [(key, value) for key, value in biases.items()]
        biases.sort(key=lambda pair: pair[0])
        biases = [value for label, value in biases]
        self.biases = biases

        # create mirror to map feature label to feature id
        mirror = {label: id for id, label in enumerate(labels)}
        self.mirror = mirror

        # establish dimensions of the feature space
        self.dimensions = len(labels)

        # post analyses attributes
        self.axes = None
        self.centers = None
        self.clusters = []
        self.kmeans = None
        self.offset = None
        self.pca = None
        self.projection = None
        self.regression = None
        self.reconstructions = None

    def cast(self, features):
        """Project a vector in original dimensions into the pca space.

        Arguments:
            features: list of tuples, feature label, feature quantity

        Returns:
            list of floats, coordinates in the pca projection

        Notes:
            1) Quicker but slightly less accurate than project method
        """

        # get components
        components = self.axes
        offset = self.offset

        # replace feature label with feature ids
        features = [(self.mirror[label], quantity) for label, quantity in features]

        # accumulate the coordinates for each component
        coordinates = []
        for component, value in zip(components, offset):

            # accumulate from each feature, beginning with the offset
            coordinate = -value
            for id, quantity in features:
                coordinate += quantity * component[id] * self.biases[id]

            # append into coordinates
            coordinates.append(coordinate)

        return coordinates

    def cluster(self, resolution=2):
        """Use Kmeans to break the pca projection into clusters.

        Arguments:
            resolution=2: integer, number of clusters

        Returns:
            None
        """

        # populate kmeans attribute
        kmeans = cluster.KMeans(n_clusters=resolution)
        kmeans.fit(self.projection)
        self.kmeans = kmeans

        # count instances of each label
        labels = kmeans.labels_
        counts = {}
        for label in labels:
            counts[label] = counts.setdefault(label, 0) + 1

        # arrange labels from biggest cluster to smallest cluster
        clusters = [key for key in counts.keys()]
        clusters.sort(key=lambda x: counts[x], reverse=True)

        # adjust cluster labels in samples instances
        members = {}
        for sample, label in zip(self, kmeans.labels_):

            # set to new index
            relabel = clusters.index(label)
            sample['cluster'] = relabel

            # create entry in members dictionary
            if relabel not in members.keys():
                members[relabel] = []

            # add sample name to members
            members[relabel].append(sample['name'])

        # create clusters attribute
        members = [(key, value) for key, value in members.items()]
        members.sort(key=lambda item: item[0])
        members = [item[1] for item in members]
        self.clusters = members

        # create centers attribute with ordering adjusted to reflect new cluster label
        centers = [self.kmeans.cluster_centers_[index] for index in clusters]
        self.centers = centers

        return None

    def decompose(self, dimensions=2):
        """Use pca to get a reduced representation of the feature space.

        Arguments:
            dimensions=2: number of pca orthogonal axes

        Returns:
            None
        """

        # begin dataset
        data = []

        # build feature vectors
        for sample in self:

            # initialize feature vector with zeroes
            vector = [0.0] * self.dimensions

            # fill out vector by entering its features
            for label, quantity in sample['features']:

                # insert feature into vector
                id = self.mirror[label]
                vector[id] = quantity * self.biases[id]

            # append into dataset
            data.append(vector)

        # populate pca attributereload
        pca = decomposition.PCA(whiten=False, n_components=dimensions)
        self.pca = pca.fit(data)

        # project samples onto the pca space
        projection = pca.transform(data)
        self.projection = projection
        for sample, coordinates in zip(self, projection):

            # add projection attribute to samples
            sample['projection'] = coordinates

        # determine projection mean
        components = self.pca.components_
        mean = self.pca.mean_
        offset = [sum([weight * value for weight, value in zip(component, mean)]) for component in components]
        self.offset = offset

        # establish axes
        axes = []
        for component in components:

            # copy each axis
            axis = []
            for coordinate in component:
                axis.append(coordinate)

            # append
            axes.append(axis)

        # establish attribute
        self.axes = axes

        return None

    def pick(self, trials=10):
        """Pick several points at random and retain the one closest to a cluster center.

        Arguments:
            trials=10: integer, number of trials from which to pick the minimum

        Returns:
            list of floats, coordinates of the point
        """

        # get brackets for random point selection
        dimensions = len(self.pca.components_)
        brackets = []
        for coordinate in range(dimensions):

            # use min and max from the sample projections
            coordinates = [sample['projection'][coordinate] for sample in self]
            minimum = min(coordinates)
            maximum = max(coordinates)
            brackets.append((minimum, maximum))

        # collect random points
        collection = []
        for trial in range(trials):

            # choose random points
            point = []
            for coordinate in range(dimensions):

                # pick a random number in that coordinate's bracket
                minimum = brackets[coordinate][0]
                maximum = brackets[coordinate][1]
                guess = minimum + rand() * (maximum - minimum)
                point.append(guess)

            # add point to collection of guesses
            collection.append(point)

        # compute distances from cluster centers for all points
        scores = []
        for point in collection:

            # compare to all cluster centers
            distances = []
            for center in self.centers:

                # compute distances to the center
                distance = sum([(point[index] - center[index]) ** 2 for index in range(dimensions)])
                distances.append(distance)

            # keep only minimum
            scores.append((min(distances), point))

        # pick point that has the smallest minimum distance
        scores.sort(key=lambda x: x[0])
        closest = scores[0][1]

        return closest

    def project(self, features, annotation=None):
        """Project a vector in original dimensions into the pca space.

        Arguments:
            features: list of tuples, feature label, feature quantity
            annotation=None: string, label for point to be annotated

        Returns:
            coordinates in the pca projection
        """

        # fill out vector
        vector = np.array([0.0] * self.dimensions)
        for label, quantity in features:
            id = self.mirror[label]
            vector[id] = quantity * self.biases[id]

        # perform projection
        vector = vector.reshape(1, -1)
        projection = self.pca.transform(vector)

        # initalize reconstructions attribute
        if self.reconstructions is None:
            self.reconstructions = []

        # add to sketch if label given
        if annotation is not None:
            self.reconstructions.append((projection[0], annotation))

        return projection[0]

    def prune(self, *clusters):
        """Prune clusters from the dataset.

        Arguments:
            *cluster: unpacked list of integers, the clusters to prune

        Returns:
            None
        """

        # pop samples objects into temporary storage
        storage = []
        while len(self) > 0:
            storage.append(self.pop())

        # prune samples
        storage = [sample for sample in storage if sample['cluster'] not in clusters]

        # pop back in
        while len(storage) > 0:
            self.append(storage.pop())

        # update the registry
        registry = {sample['name']: position for position, sample in enumerate(self)}
        self.registry = registry

        return None

    def reconstruct(self, coordinates, annotation=None):
        """Reconstruct a data point based on its position in the reduced pca analysis.

        Arguments:
            coordinates: list of floats, the coordinates in the pca projection
            annotation=None: string, annotation label for the reconstructed point

        Returns:
            list of tuples, index and magnitude of the component features sorted by magnitude
        """

        # reconstruct original feature quantities
        biases = self.biases
        reconstruction = self.pca.inverse_transform(coordinates)
        reconstruction = [(self.labels[id], quantity / biases[id]) for id, quantity in enumerate(reconstruction)]
        reconstruction.sort(key=lambda x: x[1])

        # initalize reconstructions attribute
        if self.reconstructions is None:
            self.reconstructions = []

        # create annotation label
        if annotation is None:
            annotation = 'XX'
        annotation = '______' + annotation

        # store coordinates of the reconstructed point at reconstruction attribute
        self.reconstructions.append((coordinates, annotation))

        return reconstruction

    def regress(self):
        """Perform linear regression on the pca projection or first cluster thereof.

        Arguments:
            None

        Returns:
            None
        """

        # progression
        projection = self.projection

        # trim to first cluster
        if self.kmeans:
            projection = [point for point, label in zip(projection, self.kmeans.labels_) if label == 0]

        # set up regression
        regression = linear_model.LinearRegression()
        x_points = [sample[0] for sample in projection]
        x_points = [[point] for point in x_points]
        y_points = [sample[1] for sample in projection]
        self.regression = regression.fit(x_points, y_points)

    def sketch(self, map=None, annotate=True, labels=False, hide=False):
        """Sketch a plot along the first two pca axes.

        Arguments:
            map=None: string, feature label
            annotate=True: boolean, put sample ids on the points?
            labels=False: boolean, put sample labels on points instead of ids?
            hide=False: boolean, hide the datapoints on the graph?

        Returns:
            None
        """

        # set up figure
        fig = plt.figure()

        # get points
        xs = [sample['projection'][0] for sample in self]
        ys = [sample['projection'][1] for sample in self]

        # plot map of feature if given
        if map is not None:
            resolution = 30
            xmin = min(xs)
            xmax = max(xs)
            ymin = min(ys)
            ymax = max(ys)
            xsize = (float(xmax) - float(xmin)) / resolution
            ysize = (float(ymax) - float(ymin)) / resolution
            plt.xlabel('feature: ' + map)

            # loop through x's
            for xstep in range(resolution + 1):

                # loop thourgh y's
                for ystep in range(resolution + 1):

                    # calculate map
                    x = xstep * xsize + xmin
                    y = ystep * ysize + ymin
                    id = self.registry[map]
                    degree = x * self.pca.components_[0][id] + y * self.pca.components_[1][id]

                    # plot point
                    format = 'y.'
                    if degree > -1.0:
                        format = 'yx'
                    if degree > 0.0:
                        format = 'cx'
                    if degree > 1.0:
                        format = 'c.'
                    plt.plot(x, y, format)

        # plot points
        colors = ('r', 'g', 'b', 'm', 'c', 'y')
        marker = 'o'
        formats = [color + marker for color in colors]
        for id, sample in enumerate(self):

            # set up coordinates
            projection = sample['projection']
            x = projection[0]
            y = projection[1]
            cluster = sample['cluster']
            format = cluster % len(colors)

            # plot main points and annotate if not hidden
            if hide is False:
                plt.plot(x, y, formats[format])

                # annotate?
                if annotate is True:

                    # use sample id or sample label?
                    label = str(id)
                    if labels:
                        label = self[id]['name']

                    # annotate
                    plt.annotate(xy=projection[:2], s=label)

        # plot cluster centers
        if self.kmeans is not None:
            marker = '*'
            formats = [color + marker for color in colors]
            centers = self.centers

            # plot each center
            for index, center in enumerate(centers):
                format = index % len(colors)
                plt.plot(center[0], center[1], formats[format])

        # draw reconstructed points if available
        if self.reconstructions is not None:

            # loop through all reconstructed points
            for reconstruction in self.reconstructions:

                # plot
                plt.plot(reconstruction[0][0], reconstruction[0][1], 'ko')

                # annotate
                if annotate is True:
                    plt.annotate(xy=[reconstruction[0][0], reconstruction[0][1]], s=reconstruction[1])

        # draw regression line if available
        if self.regression is not None:
            slope = self.regression.coef_[0]
            intercept = self.regression.intercept_
            xs = [-2.0, 2.0]
            ys = [x * slope + intercept for x in xs]
            plt.plot(xs, ys, 'g-')

        # show plot
        fig.show()

        return None

    def site(self, point):
        """Find the cluster center closest to the point given.

        Arguments:
            point: list of floats, the point to be compared

        Returns:
            list of floats, the location of the closest cluster center
        """

        # calculate distances
        centers = self.centers
        distances = [sum([(b - a) ** 2 for a, b in zip(point, center)]) for center in centers]

        # get closest
        ranking = [(one, two) for one, two in zip(centers, distances)]
        ranking.sort(key=lambda pair: pair[1])
        closest = ranking[0][0]

        return closest

    def survey(self, size=10):
        """Survey pca vectors.

        Arguments:
            size=10: integer, number of features in each vector to display

        Returns:
            None
        """

        # get pca components
        components = self.pca.components_

        # align with feature labels
        components = [[(self.labels[id], weight) for id, weight in enumerate(component)] for component in components]

        # sort by highest weights and print to screen
        for index, component in enumerate(components):

            # sort by absolute highest weight
            component.sort(key=lambda feature: abs(feature[1]), reverse=True)

            # print to screen
            print('%d)' % index)
            for label, weight in component[:size]:
                print(label + ': ' + str(weight))
            print(' ')

        return None

    def view(self, *clusters):
        """View the entries in a given cluster, or all clusters by default.

        Arguments:
            *cluster: unpacked tuples of cluster to print, all by default

        Returns:
            None
        """

        # sort sample names by cluster
        samples = [(sample['name'], sample['cluster']) for sample in self]
        samples.sort(key=lambda x: x[1])

        # break into seperate lists by cluster
        fragments = {}
        for sample in samples:

            # add to sublist of clusters
            label = sample[1]
            name = sample[0]
            if label in fragments.keys():
                fragments[label].append(name)

            # otherwise start new sublist
            else:
                fragments[label] = [name]

        # keep only specified clusters
        visibles = fragments.keys()
        if clusters:
            visibles = [label for label in visibles if label in clusters]

        # print all visible clusters
        visibles.sort()
        for label in visibles:
            print('cluster: %d' % label)

            # print each name
            for name in fragments[label]:
                print(name)

            # spacer
            print(' ')

        return None


# Bard class
class Bard(list):
    """Class for a song writer.

    Inherits from list.
    """

    def __init__(self, *artists):
        """Initialize a Bard as a group of artists, the influences.

        Arguments:
            *artists: unpacked tuple of strings, artist names

        Attributes:
            ambition: integer, length of line attempting to write
            auto: boolean, automatically record?
            breadth: integer, max number of samples per cluster to meditate on
            brevity: float, lower bounds of line length histogram to consider
            brocas: Grammar instance modelling word order
            capacity: integer, length of beginning string of words prior to musing
            capricious: boolean, use muse style instead of envision style for holistic line generation
            claustrum: Grammar instance modelling content and word order of each utterance
            compositions: list of songs written
            concentration: integer, number of meditation rounds per parameter
            cortex: Grammar instance modelling word frequency and length
            dimensions: integer, number of dimensions for the pca models
            diligence: integer, maximum number of positions to try while musing
            equalizer: float, left right balance when measuring distance using both claustrum and isoclaustrum
            explanation: list of strings, the explanation for thematic choices
            focus: int, number of different words to try in a slot while envisioning
            hippocampus: Lexicon instance
            holistic: boolean, use envision concept generation instead of imagine?
            imitation: float, the extent thematic associations imitate those of the influences
            introspective: boolean, track composition decisions?
            isoclaustrum: Grammar instance modelling content and backwards order of each utterance
            latitude: integer, number of best rhymes from which to pick
            limit: integer or None, number of mulling rounds applied, no limit for None
            metaphor: integer, number of words away from a thematic word to target as the likeliest
            mode: string, 'left', 'right', or 'both', which claustrums to use
            notebook: dictionary mapping word to thematic link
            objectives: list of strings, attributes under guided meditation
            obsession: integer, number of times the same thematic word my be used
            omissions: list of strings, words to omit from use
            outliers: list of integers, the indices of clusters to prune
            perceptive: boolean, true for keeping track of timepoints
            persistence: int, number of revisions to an envisioned line
            quick: boolean, True for articulate mode of word ordering and False for phrase mode
            ranges: dictionary mapping parameter names to tuples of their default ranges
            redundance: float, penalty factor for a line with a repeat of a word already used at the beginning
            resolution: integer, number of clusters
            rhyming: boolean, currently working on a rhyming line?
            ruminations: integer, number of trial ideas from which to pick
            schemes: dictionary mapping strings to strings, standard song forms to rhyme schemes
            scope: integer, number of thematic matches used for each requested
            skew: boolean, skew the pca axis based on word frequency?
            songs: list of (artist, song) tuples, the artists and songs listened to
            structure: float, weighting of word frequency when picking words
            syntax: dictionaries mapping each word or category to its dictionary of ideal category distances
            template: list of words, integers: the working line template
            timepoint: float, the latest timepoint taken
            thoroughness: integer, number of initial ideas to mull over
            verbosity: float, upper range of line length histogram to pull from
            wernickes: Grammar instance modelling word groupings
            whimsy: float, distance away from a claustrum cluster center to target
        """

        # for each artist given
        for name in artists:

            # create an instance and gather known material
            artist = Artist(name)
            artist.catalog()
            artist.gather()

            # append
            self.append(artist)

        # initiate grammar models
        self.brocas = None
        self.claustrum = None
        self.cortex = None
        self.hippocampus = None
        self.isoclaustrum = None
        self.syntax = None
        self.wernickes = None

        # initiate storage attributes
        self.ambition = None
        self.compositions = []
        self.omissions = []
        self.rhyming = False
        self.songs = []
        self.timepoint = None
        self.template = None

        # model setup parameters
        self.dimensions = 10
        self.outliers = [0]
        self.resolution = 30
        self.skew = False

        # left right balance
        self.balance = 1.0
        self.mode = 'right'

        # composition parameters
        self.redundance = 1.5
        self.ruminations = 50
        self.whimsy = 1.0

        # standard schemes
        schemes = {}
        schemes['blues'] = 'abb'
        schemes['couplet'] = 'aa'
        schemes['couplets'] = 'aa'
        schemes['drift'] = 'aaaa aaaa aa'
        schemes['limerick'] = 'aabba'
        schemes['quatrain'] = 'abcb'
        schemes['quatrains'] = 'abcb'
        schemes['rock'] = 'abcb dd'
        schemes['sonnet'] = 'abab cdcd efef gg'
        self.schemes = schemes

        # line length histogram brackets
        self.brevity = 0.0
        self.verbosity = 0.75

        # inclusion generators
        self.imitation = 0.1
        self.latitude = 15
        self.metaphor = 3
        self.obsession = 1
        self.scope = 1

        # mulling parameters
        self.limit = 2
        self.thoroughness = 10

        # brainstroming method toggles
        self.capricious = True
        self.holistic = True

        # muse method
        self.capacity = 25
        self.structure = 1.0
        self.diligence = 5

        # envision method
        self.focus = 25
        self.persistence = 15

        # imagine parameters
        self.quick = True

        # meditation parameters
        self.breadth = 20
        self.concentration = 50

        # set up parameter ranges dictionary
        ranges = {}
        ranges['balance'] = (-1.0, -0.5, 0.0, 0.5, 1.0)
        ranges['capacity'] = (20, 25, 30, 35, 40)
        ranges['diligence'] = [2, 4, 6, 8, 10, 12, 14]
        ranges['focus'] = [10, 15, 20, 25, 30]
        ranges['persistence'] = [5, 10, 15]
        ranges['structure'] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
        ranges['whimsy'] = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
        self.ranges = ranges

        # set up guided meditation parameters
        self.objectives = ['whimsy', 'capacity', 'diligence', 'structure']

        # debugging toggles
        self.introspective = False
        self.perceptive = False

        # automatically record songs?
        self.auto = True

        # initiate explanation
        self.explanation = []
        self.notebook = {}

    # instance methods
    def articulate(self, concept, notions=None):
        """Articulate a concept, finding the best word order by examining the effect of exchanging words.

        Arguments:
            concept: string, words in an order
            notions: dictionary mapping words to imposed positions

        Returns:
            (string, float) tuple, words in the best order and the tension score
        """

        # deconstruct concept into words, setting each location equal to its index
        words = concept.split()
        size = len(words)
        indices = [index for index in range(size)]
        locations = indices

        # create default notions as empty dictionary
        if notions is None:
            notions = {}

        # swap words in the concept in order to satisfy notions, and keep track of these indices
        omitted = []
        for word, location in notions.items():

            # find present location of word and the occupying word
            present = words.index(word)
            occupier = words[location]

            # swap
            words[present] = occupier
            words[location] = word

            # add location to ommitted indices
            omitted.append(location)

        # find the categories for each word
        categories = [self.cortex[self.cortex.registry[word]]['cluster'] for word in words]

        # compile ideal distance and tension information for each pair of words, weighting the tensions
        # according to available information
        # weight 0 if there is a lexicon match
        # weight 1 if there is a word-category match
        # weight 2 if there is a category-category match
        # weight 3 if the words are the same (repulsive condition)
        ideals = {}
        tensions = {}
        weights = [1.0, 0.5, 0.2, 10.0]
        for i in indices:

            # ignore equal indices
            for j in [index for index in indices if index != i]:

                # there may be an ideal distance between the pair of words given by the lexicon
                try:
                    ideals[(i, j)] = self.hippocampus[words[i]][words[j]]

                    # given the current locations of the two words, the tension is defined as the difference between
                    # this distance and the ideal distance, squared, and adjusted with a weighting
                    tensions[(i, j)] = lambda a, b, q: weights[0] * (q - (b - a)) ** 2

                    # monitor
                    if self.introspective:
                        print(words[i], words[j], ideals[(i, j)])

                # otherwise there may be a word-category relationship defined by the syntax
                except KeyError:
                    try:
                        ideals[(i, j)] = self.syntax[words[i]][categories[j]]
                        tensions[(i, j)] = lambda a, b, q: weights[1] * (q - (b - a)) ** 2

                        # monitor
                        if self.introspective:
                            print(words[i], words[j], categories[j], ideals[(i, j)])

                    # otherwise there may be a category-category relationship defined by the syntax
                    except KeyError:
                        try:
                            ideals[(i, j)] = self.syntax[categories[i]][categories[j]]
                            tensions[(i, j)] = lambda a, b, q: weights[2] * (q - (b - a)) ** 2

                            # monitor
                            if self.introspective:
                                print(words[i], categories[i], words[j], categories[j], ideals[(i, j)])

                        # otherwise there may be no ideal distance, and hence no tension
                        except KeyError:
                            ideals[(i, j)] = None
                            tensions[(i, j)] = lambda a, b, q: 0.0

                            # unless the words happen to be the same, in which case there is a repulsion
                            if words[i] == words[j]:
                                tensions[(i, j)] = lambda a, b, q: weights[3] / (b - a) ** 2

        # evaluate all tensions according to current locations
        status = {duo: tension(locations[duo[0]], locations[duo[1]], ideals[duo]) for duo, tension in tensions.items()}

        # calculate the changes in tension for each exchange until no exchange results in less tension
        better = True
        while better:

            # attempt each possible exchange between each index except those omitted
            exchanges = []
            for i in [index for index in indices if index not in omitted]:

                # and each other index
                truncation = indices.index(i)
                for j in [index for index in indices[truncation + 1:] if index not in omitted]:

                    # calculate the change in tension for the switched words
                    duo = (i, j)
                    switched = tensions[duo](locations[j], locations[i], ideals[duo])
                    change = switched - status[duo]

                    # and in the opposite order
                    duo = (j, i)
                    switched = tensions[duo](locations[i], locations[j], ideals[duo])
                    change += switched - status[duo]

                    # add the effect relative to all other words
                    for k in [index for index in indices if index != i and index != j]:

                        # compare to the first exchanged word in one order
                        duo = (i, k)
                        switched = tensions[duo](locations[j], locations[k], ideals[duo])
                        change += (switched - status[duo])

                        # and the other
                        duo = (k, i)
                        switched = tensions[duo](locations[k], locations[j], ideals[duo])
                        change += (switched - status[duo])

                        # compare to the second exchanged word in one order
                        duo = (j, k)
                        switched = tensions[duo](locations[i], locations[k], ideals[duo])
                        change += (switched - status[duo])

                        # and the other
                        duo = (k, j)
                        switched = tensions[duo](locations[k], locations[i], ideals[duo])
                        change += (switched - status[duo])

                    # store the changed value
                    exchanges.append(((i, j), change))

            # sort by most negative change
            exchanges.sort(key=lambda item: item[1])

            # update locations if the best change is negative (within float error)
            if len(exchanges) > 0:
                exchange = exchanges[0][0]
                score = exchanges[0][1]
                tolerance = -1e-14
                if score < tolerance:

                    # switch locations
                    first = exchange[0]
                    second = exchange[1]
                    updates = [(first, locations[second]), (second, locations[first])]
                    for index, location in updates:
                        locations[index] = location

                    # update tensions
                    for i in indices:
                        for j in [index for index in indices if index != i]:

                            # do update if affected
                            if i in exchange or j in exchange:

                                # update status
                                duo = (i, j)
                                status[duo] = tensions[duo](locations[i], locations[j], ideals[duo])

                # otherwise no exchange is better
                else:
                    better = False

            # otherwise no exchange is possible
            else:
                better = False

        # calculate average tension per word, squared
        tension = sum(status.values())
        tension /= float(size) ** 2

        # arrange words according to new locations
        arrangement = zip(words, locations)
        arrangement.sort(key=lambda item: item[1])
        idea = [item[0] for item in arrangement]
        idea = ' '.join(idea)

        # monitor
        if self.introspective:
            print(idea, tension)
            pass

        return idea

    def augment(self, name):
        """Augment the current collection by adding an artist.

        Arguments:
            name: string, name of the artist

        Returns:
            None
        """

        # create an instance and gather known material
        artist = Artist(name)
        artist.catalog()
        artist.gather()

        # append
        self.append(artist)

    def brainstorm(self, *notions):
        """Brainstorm a lyric.

        Arguments:
            *notions: unpacked tuple of strings and integers

        Returns:
            strings, a lyric
        """

        # distill words to be included
        inclusions = [element for element in notions if element == str(element)]

        # monitor:
        if self.introspective:
            #print inclusions
            pass

        # distill the given notions into a mapping
        notions = zip(notions[:-1], notions[1:])
        notions = [(first, second) for first, second in notions if first == str(first)]
        notions = [(first, second) for first, second in notions if second != str(second)]
        notions = dict(notions)

        # envision an idea all at once
        if self.holistic:

            # either condensing a random string of words
            if self.capricious:
                idea = self.muse(inclusions, notions)

            # or rotating and swaping
            else:
                idea = self.envision(inclusions, notions)

        # otherwise imagine a concept
        else:
            concept = self.imagine(*inclusions)

            # change negative indices to positive indices
            length = len(concept.split())
            for word, position in notions.items():

                # change negative to positives
                if position < 0:
                    notions[word] = length + position

            # articulate the concept in more logical structure
            # using learned contexts and categories
            if self.quick:
                idea = self.articulate(concept, notions)

            # or by using claustrum space
            else:
                idea = self.phrase(concept, notions)

        return idea

    def categorize(self, resolution=15):
        """Categorize words based on frequency and contextual clusters.

        Arguments:
            resolution=15: integer, number of categories

        Returns:
            None
        """

        # build two dimensional feature space
        # first feature is the frequency of the word
        # second feature is the context category
        brocas = self.brocas
        frequencies = self.hippocampus.frequencies
        data = []
        for sample in brocas:
            name = sample['name']
            frequency = frequencies[name]
            cluster = sample['cluster']
            features = [('frequency', frequency), ('cluster', cluster), ('length', len(name))]
            data.append((name, features))

        # generate cortex attribute
        cortex = Grammar(data)
        self.cortex = cortex

        # perform machine learning
        self.cortex.decompose(3)
        self.cortex.cluster(resolution)

        return None

    def clear(self):
        """Clear the working line template.

        Arguments:
            None

        Returns:
            None
        """

        # clear template, ambition, timepoint, and explanation
        self.template = None
        self.ambition = None
        self.timepoint = None

        return None

    def compose(self, length=1, scheme='aa', *themes):
        """Compose a song.

        Arguments:
            length=1: integer, number of verses
            scheme='aa': string, rhyme scheme of each verse
            *themes: unpacked tuple of strings, thematic words

        Returns:
            list of lists of strings, the lyrics in each verse
        """

        # clear explanation
        self.explanation = []
        self.notebook = {}

        # set number of ideas to try
        trials = self.ruminations

        # set number of best rhyming words to consider
        latitude = self.latitude

        # set distance from a cluster center to target
        whimsy = self.whimsy

        # score words based on relationship to given themes
        relations = []
        if len(themes) > 0:
            relations = self.consider(*themes)

        # monitor
        if self.introspective:
            #print relations
            #print ' '
            pass

        # check for standard schemes
        schemes = self.schemes
        if scheme in schemes.keys():
            scheme = schemes[scheme]

        # begin tracker
        lines = length * len(''.join(scheme.split()))
        print('composing (%d)' % lines)

        # begin song
        song = []
        beginnings = {}
        obsession = self.obsession
        recall = {word: 0 for word, score in relations}
        for count in range(length):

            # monitor
            if self.introspective or self.perceptive:
                print(' ')
                print('verse %d:' % (count))

            # begin verse
            verse = []
            rhymes = {}
            for letter in scheme:

                # append a blank lyric if the letter is a blank
                if letter == ' ':
                    verse.append(' ')
                    self.explanation.append(' ')
                    continue

                # clear line template, ambition, and reset timepoint
                self.clear()

                # beginning timing
                if self.perceptive:
                    self.mark()

                # monitor
                if self.introspective or self.perceptive:
                    print(' ')
                    print('line ' + letter + ':')

                # try to generate a rhyming line if letter has already been used
                try:

                    # generate rhyme scores based on most recently used word:
                    rhymers = rhymes[letter]
                    rhymer = rhymers[-1]
                    scores = []
                    for word in self.hippocampus.vocabulary:

                        # exclude the same word and omitted words and words with apostrophes
                        if word not in rhymers and word not in self.omissions and "'" not in word:

                            # test the rhyme distance between the rhymer and each other word
                            scores.append((word, self.rhyme(rhymer, word)))

                    # keep the top possibilities
                    scores.sort(key=lambda score: score[1])
                    scores = scores[:latitude]

                    # multiply the rhyme scores by a random number and rank by the result
                    ranks = [(word, score * rand()) for word, score in scores]
                    ranks.sort(key=lambda x: x[1], reverse=True)

                    # use the first rank as the last word in an idea
                    inclusion = ranks.pop()[0]

                    # add explanation entry
                    entry = '%s: rhymes with %s' % (inclusion, rhymer)
                    self.explanation.append(entry)

                    # timepoint
                    if self.perceptive:
                        print('finding rhyme:')
                        self.mark()

                    # brainstorm
                    ideas = []
                    for trial in range(trials):

                        # monitor
                        if self.introspective or self.perceptive:
                            print(' ')
                            print('rumination %d of %d:' % (trial + 1, trials))

                        # use inclusion as the last word in an idea
                        ideas.append(self.brainstorm(inclusion, -1))

                    # project each idea into the claustrum spaces
                    scores = [(idea, self.reflect(idea)) for idea in ideas]
                    scores.sort(key=lambda pair: abs(pair[1] - whimsy))

                    # monitor
                    if self.introspective:
                        print(' ')
                        for score in scores:
                            print(score[1], score[0])

                    # mull over the top few
                    self.rhyming = True
                    best = self.thoroughness
                    scores = scores[:best]
                    scores = [self.mull([inclusion], *score) for score in scores]

                    # timepoint:
                    if self.perceptive:
                        print(' ')
                        print('mulling:')
                        self.mark()

                # otherwise seed a line with themtaic matches
                except KeyError:

                    # find a thematic inclusion if themes are given
                    inclusions = []
                    if len(relations) > 0:

                        # thin out relations by words that have already been used an obsession's number of times
                        relations = [relation for relation in relations if recall[relation[0]] < obsession]

                        # multiply each thematic score by a random number and rank
                        ranks = [(word, relation * rand()) for word, relation in relations]
                        ranks.sort(key=lambda rank: rank[1], reverse=True)

                        # monitor
                        if self.introspective:
                            #print ranks
                            #print ' '
                            pass

                        # include two inclusions in the line
                        for episode in range(2):
                            inclusion = ranks[episode][0]
                            inclusions.append(inclusion)
                            recall[inclusion] += 1

                            # add notebook entry for inclusion to explanations
                            self.explanation.append(self.notebook[inclusion])

                        # monitor
                        if self.introspective:
                            print(inclusions)
                            pass

                        # timepoint
                        if self.perceptive:
                            print('finding thematic choices:')
                            self.mark()

                        # brainstorm
                        ideas = []
                        for trial in range(trials):

                            # monitor
                            if self.introspective or self.perceptive:
                                print(' ')
                                print('rumination %d of %d:' % (trial + 1, trials))

                            # brainstorm
                            ideas.append(self.brainstorm(*inclusions))

                        # project each idea into the claustrum spaces
                        scores = [(idea, self.reflect(idea)) for idea in ideas]
                        scores.sort(key=lambda pair: abs(pair[1] - whimsy))

                        # monitor
                        if self.introspective:
                            print(' ')
                            for score in scores:
                                print(score[1], score[0])

                        # mull over the top few
                        self.rhyming = False
                        best = self.thoroughness
                        scores = scores[:best]
                        scores = [self.mull(inclusions, *score) for score in scores]

                        # timepoint
                        if self.perceptive:
                            print(' ')
                            print('mulling:')
                            self.mark()

                # determine first line redundancy adjustments
                redundance = self.redundance
                scores.sort(key=lambda pair: pair[1])
                adjustments = [0.0 for score in scores]
                for index, score in enumerate(scores):
                    last = score[0].split()[0]
                    if last in beginnings.keys():
                        adjustments[index] = redundance * beginnings[last]

                # apply factors and sort
                scores = [(score[0], score[1] + adjustments[index] + whimsy) for index, score in enumerate(scores)]
                scores.sort(key=lambda pair: abs(whimsy - pair[1]), reverse=True)

                # grab the last line and update the beginnings record
                lyric = scores.pop()[0]
                last = lyric.split()[0]
                if last not in beginnings.keys():
                    beginnings[last] = 0
                beginnings[last] += 1

                # create rhyming entry and update
                if letter not in rhymes.keys():
                    rhymes[letter] = []
                rhymes[letter].append(lyric.split()[-1])

                # add to verse
                verse.append(lyric)

                # timepoint:
                if self.perceptive:
                    print(' ')
                    print('finishing line:')
                    self.mark()

                # update tracker
                print('.')

            # append the verse
            song.append(verse)

            # add explanation space
            self.explanation.append(' ')

        # put the song into the compositions
        print(' ')
        print(' ')
        self.compositions.append(song)

        return None

    def conceptualize(self, dimensions=4, resolution=15):
        """Develop the wernickes attribute with a Grammar instance modelling word groupings.

        Arguments:
            dimensions=4: integer, number of pca component axes
            resolution=15: integer, number of concept clusters

        Returns:
            None
        """

        # map words to their category numbers
        mapping = {sample['name']: sample['cluster'] for sample in self.cortex}

        # reinterpret each lyric as a tally of category bins
        data = []
        for lyric in self.hippocampus.lyrics:

            # make into category tallies
            features = self.diagram(lyric)
            data.append((lyric, features))

        # create grammar instance
        self.wernickes = Grammar(data)

        # perform machine learning analysis
        self.wernickes.decompose()
        self.wernickes.cluster(resolution)
        self.wernickes.prune(resolution - 1)
        self.wernickes.decompose(dimensions)
        self.wernickes.cluster(resolution)

        return None

    def consider(self, *themes):
        """Consider words most closely related to the given themes to seed an idea.

        Arguments:
            *themes: unpacked tuple of strings, words given as themes

        Returns:
            list of (word, float) tuples, the word and its match score
        """

        # find closest matching words for each theme, excluding omissions
        vocabulary = [word for word in self.hippocampus.vocabulary if word not in self.omissions]
        matches = []
        for theme in themes:

            # score each word in the vocabulary according to closeness to the theme, prioritizing smaller words if equal
            scores = [(word, self.gauge(word, theme)) for word in vocabulary]
            scores.sort(key=lambda score: len(score[0]))
            scores.sort(key=lambda score: score[1], reverse=True)

            # add the first members to the matches
            scope = self.scope
            matches += [score[0] for score in scores[:scope]]

        # multiply list of themes to match extent
        themes = [[theme] * scope for theme in themes]
        themes = [member for theme in themes for member in theme]

        # print themes
        print(' ')
        string = 'themes: '
        for match, theme in zip(matches, themes):
            string += match
            if theme != match:
                string += '(' + theme + ')'
            string += ', '
        print(string[:-2])

        # for each word in the vocabulary, calculate a thematic score
        scores = []
        notebook = {}
        songs = self.songs
        mapping = self.hippocampus.mapping
        omissions = self.omissions
        frequencies = self.hippocampus.frequencies
        for word in self.hippocampus.vocabulary:

            # abort if word is in omissions
            if word in omissions:
                continue

            # examine against each thematic match
            score = 0.0
            chunk = 0.0
            for match in matches:

                # examine each song
                zipper = zip(mapping[word], mapping[match], songs)
                for locations, positions, info in zipper:

                    # calculate a score based on frequency weighting the distance between occurences
                    try:
                        height = 0.1 / float(frequencies[word])

                    # unless the frequency is zero
                    except ZeroDivisionError:
                        height = 0

                    # calculate an exponential based on the distance between the words
                    imitation = self.imitation
                    metaphor = self.metaphor
                    for location in locations:
                        for position in positions:

                            # use a negative expoential,  c * exp^ -k abs(abs(a - b) - offset))
                            # to center around an offset's position away from the word in question
                            subscore = height * exp(-imitation * abs(abs(position - location) - metaphor))
                            score += subscore

                            # evaluate subscore and modify notebook
                            if subscore > chunk:
                                chunk = subscore

                                # create entry string
                                entry = '%s: related to (%s) from %s by %s' % (word, match, info[1], info[0])
                                notebook[word] = entry

            # append score
            scores.append((word, score))

        # set notebook
        self.notebook = notebook

        # sort scores, highest first
        scores.sort(key=lambda item: item[1], reverse=True)

        return scores

    def contextualize(self, dimensions=4, resolution=15):
        """Set up a Grammar instance to analyze the contextual situation of each word.

        Arguments:
            dimensions=4: integer, mumber of pca component axes
            resolution=15: integer, number of context clusters

        Returns:
            None
        """

        # build dataset of all words and their contexts
        data = []
        frequencies = self.hippocampus.frequencies
        for word in self.hippocampus.vocabulary:

            # build sparse vector from dictionary of context words and instances of their distances
            features = self.hippocampus[word].items()

            # create the data
            data.append((word, features))

        # create grammar instance
        self.brocas = Grammar(data, frequencies)

        # perform machine learning analysis
        self.brocas.decompose(dimensions)
        self.brocas.cluster(resolution)

        return None

    def diagram(self, lyric):
        """Diagram the lyric, breaking it into a tally of word categories.

        Arguments:
            lyric: string, a lyric line

        Returns:
            list of (integer, integer) tuples, word category and tally
        """

        # analyze the word categories of the lyric
        diagram = {}
        for word in lyric.split():

            # retrieve category
            number = self.cortex.registry[word]
            category = self.cortex[number]['cluster']

            # add to tall in diagram
            diagram[category] = diagram.setdefault(category, 0) + 1

        # reform into ordered list
        diagram = diagram.items()
        diagram.sort(key=lambda x: x[1])

        return diagram

    def envision(self, inclusions=None, notions=None):
        """Imagine a concept by rounds of swapping words in and around.

        Arguments:
            inclusions=Nones: list of strings, words to be included
            notions=None: diction mapping indices to words

        Returns:
            string, list of words
        """

        # set defaults
        if inclusions is None:
            inclusions = []
        if notions is None:
            notions = {}

        # establish target line length
        self.intend()

        # convert negative indices to positive indices
        ambition = self.ambition
        for word, index in notions.items():
            if index < 0:
                notions[word] = index + ambition

        # subtract by number of inclusions
        voidspace = ambition - len(inclusions)

        # fill out a concept with random words, but ignore inclusions and omissions
        words = []
        omissions = self.omissions
        vocabulary = [word for word in self.hippocampus.vocabulary if word not in inclusions + omissions]
        for trial in range(voidspace):
            words.append(choice(vocabulary))

        # add inclusions
        words += inclusions

        # randomize order
        words.sort(key=lambda word: rand())

        # monitor
        if self.introspective:
            print(' '.join(words))

        # apply notions
        for word, position in notions.items():

            # switch words to satisfy notions
            occupier = words[position]
            hole = words.index(word)
            words[hole] = occupier
            words[position] = word

        # monitor
        if self.introspective:
            print(' '.join(words))

        # project the words into claustrum space
        if self.balance < 1.0:
            features = self.parse(words)
            casting = self.claustrum.cast(features)
            #casting = self.claustrum.project(features)

            # monitor
            if self.introspective:
                print(casting)

            # find the distance to each center
            centers = self.claustrum.centers
            distances = [sum([(b - a) ** 2 for a, b in zip(casting, center)]) for center in centers]

            # get the center at the closest distance
            ranking = zip(centers, distances)
            ranking.sort(key=lambda pair: pair[1])
            closest = ranking[0][0]

            # monitor
            if self.introspective:
                print(ranking)
                print(' '.join(words))
                print(closest)

        # project the words into isoclaustrum space
        if self.balance > -1.0:
            features = self.isoparse(words)
            isocasting = self.isoclaustrum.cast(features)
            #isocasting = self.isoclaustrum.project(features)

            # find the distance to each center
            centers = self.isoclaustrum.centers
            distances = [sum([(b - a) ** 2 for a, b in zip(isocasting, center)]) for center in centers]

            # get the center at the closest distance
            ranking = zip(centers, distances)
            ranking.sort(key=lambda pair: pair[1])
            isoclosest = ranking[0][0]

            # monitor
            if self.introspective:
                print(' '.join(words))
                print(isoclosest)

        # for multiple rounds...
        rounds = self.persistence
        attempts = self.focus
        whimsy = self.whimsy
        for round in range(rounds):

            # monitor
            if self.introspective:
                window = ' '.join(words)
                print('   ' + window)
                self.posit(window)

            # pick a word at a continually incrememting position
            availables = [word for word in words if word not in inclusions]
            swap = availables[round % len(availables)]
            position = words.index(swap)

            # pick a list of words to exchange
            pile = [word for word in vocabulary if word not in words]
            exchanges = [choice(pile) for attempt in range(attempts)]

            #print swap, exchanges
            #print ' '

            # measure the distances in claustrum and isoclaustrum space for each exchange
            distances = []
            arrangement = words[:]
            for swapper in [swap] + exchanges:

                # rotate in word
                arrangement[position] = swapper

                # tare distances
                distance = 0.0
                isodistance = 0.0

                # measure distance in claustrum space
                if self.balance < 1.0:
                    features = self.parse(arrangement)
                    casting = self.claustrum.cast(features)
                    #casting = self.claustrum.project(features)
                    distance = sum([(b - a) ** 2 for a, b in zip(casting, closest)])

                # measure distance in isoclaustrum space
                if self.balance > -1.0:
                    features = self.isoparse(arrangement)
                    isocasting = self.isoclaustrum.cast(features)
                    #isocasting = self.claustrum.project(features)
                    isodistance = sum([(d - c) ** 2 for c, d in zip(isocasting, isoclosest)])

                # calculate distance
                distances.append((swapper, distance + isodistance))

            # find the closest and make the exchange
            distances.sort(key=lambda pair: abs(pair[1] - whimsy))
            words[position] = distances[0][0]

            #print distances[0][1]

            #print ' '
            #print ' '.join(words)
            #print ' '

            # reduce indices that are not in notions or refer to inclusions
            possibles = [index for index, word in enumerate(words) if index not in notions.values()]
            possibles = [index for index in possibles if words[index] not in inclusions]

            # pick index at iterating index
            pick = possibles[round % len(possibles)]

            # try swapping with every other valid index
            distances = []
            for other in possibles:

                # make exchange
                arrangement = words[:]
                first = arrangement[pick]
                second = arrangement[other]
                arrangement[pick] = second
                arrangement[other] = first

                # tare distances
                distance = 0.0
                isodistance = 0.0

                # measure distance in claustrum space
                if self.balance < 1.0:
                    features = self.parse(arrangement)
                    casting = self.claustrum.cast(features)
                    #casting = self.claustrum.project(features)
                    distance = sum([(b - a) ** 2 for a, b in zip(casting, closest)])

                # measure distance in isoclaustrum space
                if self.balance > -1.0:
                    features = self.isoparse(arrangement)
                    isocasting = self.isoclaustrum.cast(features)
                    #isocasting = self.isoclaustrum.project(isofeatures)
                    isodistance = sum([(d - c) ** 2 for c, d in zip(isocasting, isoclosest)])

                # calculate distance
                distances.append((arrangement, distance + isodistance))

            # find the closest and make the exchange
            distances.sort(key=lambda pair: abs(pair[1] - whimsy))
            words = distances[0][0]

        # monitor
        if self.introspective:
            window = ' '.join(words)
            print('   *' + window)
            self.posit(window)

            #print ' '.join(words)

        # form idea
        idea = ' '.join(words)

        return idea

    def equalize(self, songs=None):
        """Equalize the number of selected songs amongst the artists.

        Arguments:
            songs=None: integer, number of songs to keep from each artist

        Returns:
            None
        """

        # find minimum number of songs
        size = min([len(artist) for artist in self])

        # check to see if given size is smaller
        if songs:
            if songs < size:
                size = songs

        # randomly select songs for each artist
        for artist in self:

            # attach random number to each index and sort
            indices = [(index, rand()) for index, song in enumerate(artist)]
            indices.sort(key=lambda pair: pair[1])
            indices = [index for index, score in indices]

            # select all songs in the first part
            for index in indices[:size]:
                artist[index].selected = True

            # deslect all songs in second part
            for index in indices[size:]:
                artist[index].selected = False

        return None

    def exclude(self, *selections):
        """Exclude songs from study.

        Arguments:
            *selections: unpacked tuple of strings and integers

        Returns:
            None
        """

        # make into list
        selections = list(selections)

        # by default, select all
        if len(selections) < 1:

            # change each status
            for artist in self:
                for song in artist:
                    song.selected = False

            return None

        # put by default the first artists' name if selections begins with a number
        if str(selections[0]).isalpha() == False:
            selections = [self[0].name] + selections

        # mark the positions of alpha strings
        positions = [index for index, token in enumerate(selections) if str(token).isalpha()]

        # if all of the positions are alpha strings, assume they are song titles
        if len(positions) == len(selections):

            # check each selection against all song titles
            songs = [song for artist in self for song in artist]
            for selection in selections:

                # score each by distance
                scores = [(song, self.gauge(selection, song.name)) for song in songs]
                scores.sort(key=lambda pair: pair[1], reverse=True)

                # set the best match to selected
                best = scores[0][0]
                best.selected = False

                # print to screen
                print(best.name)
                print(' ')

            return None

        # otherwise, account for final index and zip off into pairs
        positions.append(len(selections))
        pairs = zip(positions[:-1], positions[1:])
        choices = [selections[first: second] for first, second in pairs]
        choices = [list(choice) for choice in choices]

        # find index of each band name
        artists = []
        for choice in choices:

            # find closest matches to artists' names
            name = choice[0]
            scores = [(index, artist.gauge(artist.name, name)) for index, artist in enumerate(self)]

            # retain top match
            scores.sort(key=lambda score: score[1], reverse=True)
            artists.append(scores[0][0])

        # for each choice
        for artist, choice in zip(artists, choices):

            # change the status for every song
            for song in choice[1:]:

                # but skip non existent indices
                try:
                    self[artist][song].selected = False
                except IndexError:
                    pass

        return None

    def explain(self):
        """Explain the thematic relationships in the most recently composed song.

        Arguments:
            None

        Returns:
            None
        """

        # print explanation
        print(' ')
        explanation = self.explanation
        for line in explanation:
            print(line)

        return None

    def forget(self, *words):
        """Omit certain words from using in songs by adding to the omissions attribute.

        Arguments:
            *words: unpacked tuple of strings

        Returns:
            None
        """

        # add to omissions
        for word in words:
            self.omissions.append(word)

        return None

    @staticmethod
    def gauge(trial, target, window=4):
        """Gauge the closeness of a trial string to a target string by counting common letter groupings

        Arguments:
            trial: string
            target: string
            window=4: integer, length of letter groupings to examine

        Returns:
            integer, relatedness score
        """

        # begin with largest bracket
        bracket = window

        # whittle bracket down to one
        score = 0.0
        while bracket > 0:

            # break target into groupings
            partials = []
            partial = target
            for round in range(bracket):
                partials.append(partial)
                partial = partial[1:]

            # remerge as groupings
            groupings = zip(*partials)
            groupings = [''.join(group) for group in groupings]

            # count occurrences in trial
            occurrences = [group for group in groupings if group in trial]

            # weight by inverse bracket size and add to score
            weight = 10 ** (bracket - window)
            score += len(occurrences) * weight

            # decrement bracket
            bracket -= 1

        return score

    def generalize(self):
        """Generalize the contextual relations amongst words and word categories.

        Arguments:
            None

        Returns:
            None
        """

        # combine words and categories
        words = [word for word in self.hippocampus.vocabulary]
        categories = [item[0] for item in enumerate(self.cortex.clusters)]
        entries = words + categories
        syntax = {entry: {category: {} for category in categories} for entry in entries}

        # inspect each word and its category
        for wordi in self.hippocampus.vocabulary:
            categoryi = self.cortex[self.cortex.registry[wordi]]['cluster']

            # and each other word and its category
            for wordii, distance in self.hippocampus[wordi].items():
                categoryii = self.cortex[self.cortex.registry[wordii]]['cluster']

                # get the frequency of the second word
                frequency = self.hippocampus.frequencies[wordii]

                # add the frequency to the syntax entries if the categories are different
                if categoryi != categoryii:
                    syntax[wordi][categoryii][distance] = syntax[wordi][categoryii].setdefault(distance, 0.0) + frequency
                    syntax[categoryi][categoryii][distance] = syntax[categoryi][categoryii].setdefault(distance, 0.0) + frequency

        # define compression function for choosing the most highly represented distance from a dictionary
        # of contextual instances
        def compress(tallies):

            # sort items by size
            tallies = tallies.items()
            tallies.sort(key=lambda item: item[1])

            # return the mode, or 0 if missing
            try:
                mode = tallies.pop()[0]
            except IndexError:
                mode = 0

            return mode

        # define trim function to remove zero entries
        def trim(relations):

            # remove zero entries
            relations = relations.items()
            relations = {category: distance for category, distance in relations if distance != 0}

            return relations

        # compress syntaxes to the modes, and remove zeros
        syntax = syntax.items()
        syntax = {entry[0]: {category: compress(tallies) for category, tallies in entry[1].items()} for entry in syntax}
        syntax = {entry[0]: trim(entry[1]) for entry in syntax.items()}

        # populate attribute
        self.syntax = syntax

        return None

    def grab(self, *songs):
        """Grab a song from each artist from the web.

        Arguments:
            *s: unpacked tuple of songtitles, in same order as artists

        Returns:
            None
        """

        # pad s with blanks
        while len(songs) < len(self):
            songs += ' ',

        # zip song titles and bands
        zipper = zip(self, songs)

        # grab a song from each
        for artist, song in zipper:
            artist.grab(song)

        return None

    def imagine(self, *inclusions):
        """Imagine a concept from the working line template.

        Arguments:
            *inclusions: unpacked tuples of strings, words to be included

        Returns:
            string, list of words
        """

        # make a template if there is not one already
        if self.template is None:
            self.suppose(*inclusions)

        # split instances from the template
        template = self.template
        instances = [member for member in template if member not in inclusions]

        # create concept, beginning with the inclusions
        concept = list(inclusions)
        for category in instances:

            # pick a member of the category at random, excluding omitted words
            cluster = self.cortex.clusters[category]
            cluster = [member for member in cluster if member not in self.omissions]
            word = choice(cluster)
            concept.append(word)

        # filter out duplicates
        concept = list(set(concept))

        # make into string
        concept = ' '.join(concept)

        # monitor
        if self.introspective:
            print(' ')
            print(concept)
            pass

        return concept

    def include(self, *selections):
        """Include songs for study.

        Arguments:
            *selections: unpacked tuple of strings and integers

        Returns:
            None
        """

        # make into list
        selections = list(selections)

        # by default, select all
        if len(selections) < 1:

            # change each status
            for artist in self:
                for song in artist:
                    song.selected = True

            return None

        # put by default the first artists' name if selections begins with a number
        if str(selections[0]).isalpha() == False:
            selections = [self[0].name] + selections

        # mark the positions of alpha strings
        positions = [index for index, token in enumerate(selections) if str(token).isalpha()]

        # if all of the positions are alpha strings, assume they are song titles
        if len(positions) == len(selections):

            # check each selection against all song titles
            songs = [song for artist in self for song in artist]
            for selection in selections:

                # score each by distance
                scores = [(song, self.gauge(selection, song.name)) for song in songs]
                scores.sort(key=lambda pair: pair[1], reverse=True)

                # set the best match to selected
                best = scores[0][0]
                best.selected = True

                # print to screen
                print(best.name)
                print(' ')

            return None

        # otherwise, account for final index and zip off into pairs
        positions.append(len(selections))
        pairs = zip(positions[:-1], positions[1:])
        choices = [selections[first: second] for first, second in pairs]
        choices = [list(choice) for choice in choices]

        # find index of each band name
        artists = []
        for choice in choices:

            # find closest matches to artists' names
            name = choice[0]
            scores = [(index, artist.gauge(artist.name, name)) for index, artist in enumerate(self)]

            # retain top match
            scores.sort(key=lambda score: score[1], reverse=True)
            artists.append(scores[0][0])

        # for each choice
        for artist, choice in zip(artists, choices):

            # change the status for every song
            for song in choice[1:]:

                # but skip non existent indices
                try:
                    self[artist][song].selected = True
                except IndexError:
                    pass

        return None

    def intend(self, lower=None, upper=None):
        """Establish a target line length.

        Arguments:
            lower=None: float, lower bounds of length histogram to pull from
            upper=None: float, upper bounds of length histogram to pull from

        Returns:
            None
        """

        # set default line length parameters
        if lower is None:
            lower = self.brevity
        if upper is None:
            upper = self.verbosity

        # query the histogram for a line length if not already done so
        if self.ambition is None:

            # keep only the middle 60% of the histogram to avoid very long lines
            histogram = [bar for bar in self.hippocampus.histogram]
            histogram = [bar for bar in histogram if bar[1] > lower and bar[1] < upper]
            histogram.reverse()

            # pick a bin at random
            rando = rand()
            measurement = histogram[0][0]
            for category, weight in histogram:
                if rando < weight:
                    measurement = category

            # set ambition
            self.ambition = measurement

        return None

    def isoparse(self, words, target=None):
        """Parse a list of words into a feature space vector, scaled based on a target length.

        Arguments:
            words: list of strings, words in the lyric
            target=None: integer, scale the vector based on a target length

        Return:
            list of (string, float) tuples, the features and their positions
        """

        # set default target to current length
        length = len(words)
        if target is None:
            target = length

        # scale positions to target length
        scaling = float(target) / float(length)
        features = [(word, (((index - length) + 1) * scaling) - 1) for index, word in enumerate(words)]

        return features

    def listen(self):
        """Listen to all the lyrics, creating the lexicon attribute.

        Arguments:
            None

        Returns:
            None
        """

        # kludge every song by every artist
        transcript = []
        songs = []
        for artist in self:

            # and accumulate its lyrics into a transcript
            for song in artist:

                # only study selected songs
                if song.selected:

                    # add to list of songs
                    songs.append((artist.name, song.name))

                    # avoid altering original instance
                    song = song.copy()

                    # filter out blank lines, /r, and bracketted lines
                    song.keep(lambda line: len(line) > 0)
                    song.keep(lambda line: line != '\r')
                    song.keep(lambda line: '[' not in line)
                    song.keep(lambda line: ']' not in line)

                    # change to lower case and remove duplicates
                    song.apply(lambda line: line.lower())
                    song.skim()

                    # filter each line
                    song.apply(lambda line: Song._filter(line))

                    # compact hyphens
                    song.apply(lambda line: line.replace('-', ''))

                    # eliminate punctuation
                    punctuation = [',', '?', '!', '.', '(', ')', '*', '"', ';', ':', '&quot', '&', '_', '/', '+']
                    for symbol in punctuation:
                        song.apply(lambda line: line.replace(symbol, ' '))

                    # add lines into a set for the song
                    lyrics = []
                    for line in song:
                        lyric = line.split()
                        lyric = ' '.join(lyric)
                        lyrics.append(lyric)

                    # add to the transcript
                    transcript.append(lyrics)

        # establish songs
        self.songs = songs

        # form lexicon
        self.hippocampus = Lexicon(transcript)

        return None

    def mark(self):
        """Take a timepoint and print the time passed since last timepoint.

        Arguments:
            None

        Returns:
            None
        """

        # take initial point if None taken yet
        if self.timepoint is None:
            self.timepoint = clock()

        # otherwise print the difference
        else:
            timepoint = clock()
            print(timepoint - self.timepoint)
            self.timepoint = timepoint

        return None

    def meditate(self, parameter, *attempts):
        """Meditate on a parameter, brainstorming for several cycles at each attempted value and comparing to samples.

        Arguments:
            parameter: string, name of the parameter
            *attempts: unpacked tuple of values for the parameter

        Returns:
            None

        Notes:
            The attempted value with the closest average distances to lyrics in the database will be set.
        """

        # start monitor
        print(' ')
        print('meditating on %s:' % parameter)

        # establish default values if unspecified
        if len(attempts) < 1:
            attempts = self.ranges[parameter]

        # pick a subset of samples from claustrum space
        breadth = self.breadth
        clusters = self.claustrum.clusters
        registry = self.claustrum.registry
        samples = []
        for group in clusters:

            # sort by random number
            group.sort(key=lambda member: rand())

            # append top few to samples
            for name in group[:breadth]:

                # retrieve sample id from registry
                tag = registry[name]
                samples.append(self.claustrum[tag])

        # pick a subset of samples from isoclaustrum space
        breadth = self.breadth
        clusters = self.isoclaustrum.clusters
        registry = self.isoclaustrum.registry
        isosamples = []
        for group in clusters:

            # sort by random number
            group.sort(key=lambda member: rand())

            # append top few to samples
            for name in group[:breadth]:

                # retrieve sample id from registry
                tag = registry[name]
                isosamples.append(self.isoclaustrum[tag])

        # for each attempted value...
        scores = []
        cycles = self.concentration
        for attempt in attempts:

            # monitor
            print(str(attempt) + ':')
            print('(' + str(cycles) + ')',)

            # set the parameter to the attempt
            self.__setattr__(parameter, attempt)

            # brainstorm for several cycles
            ideas = []
            results = []
            balance = self.balance
            for cycle in range(cycles):

                # brainstorm
                idea = self.brainstorm()

                # compare to all samples in claustrum subset
                distance = 0.0
                if balance < 1.0:

                    # find the projection
                    features = self.parse(idea.split())
                    casting = self.claustrum.cast(features)

                    # calculate the distances to the projection of each sample and pick the smallest
                    distances = [sum([(b - a) ** 2 for a, b in zip(casting, sample['projection'])]) for sample in samples]
                    distances.sort()
                    distance += distances[0]
                    pass

                # project into isoclaustrum space
                isodistance = 0.0
                if balance > -1.0:

                    # find the projection
                    features = self.isoparse(idea.split())
                    casting = self.isoclaustrum.cast(features)

                    # calculate the distances to the projection of each sample and pick the smallest
                    distances = [sum([(b - a) ** 2 for a, b in zip(casting, sample['projection'])]) for sample in isosamples]
                    distances.sort()
                    isodistance += distances[0]
                    pass

                # determine claustrum, isoclaustrum balance
                if balance < 0:
                    left = -balance * 0.5 + 0.5
                    right = 1.0 - left
                else:
                    right = balance * 0.5 + 0.5
                    left = 1.0 - right

                # calculate score
                result = left * distance + right * isodistance
                results.append(result)

                # monitor
                print('.',)

            # append the average
            score = float(sum(results)) / len(results)
            scores.append(score)

            # close monitor
            print('')
            print(score)

        # zip and set the parameter
        ranking = zip(attempts, scores)
        ranking.sort(key=lambda pair: pair[1])
        self.__setattr__(parameter, ranking[0][0])

        # monitor
        print(parameter + ': ' + str(ranking[0][0]))
        print(' ')

        return None

    def metatate(self, *parameters):
        """Meditate on several parameters until convergence is reached.

        Arguments:
            *parameters: unpacked tuple of parameter names

        Returns:
            None
        """

        # default parameters
        if len(parameters) < 1:
            parameters = self.objectives

        # establish current status values
        status = [self.__getattribute__(parameter) for parameter in parameters]

        # main loop
        rounds = 3
        recording = [status]
        for round in range(rounds):

            # try each parameter
            for parameter in parameters:

                # monitor
                print(str(round) +'): ')

                # retrieve present value and standard values
                present = self.__getattribute__(parameter)
                standards = self.ranges[parameter]

                # compare present to each standard
                distances = [(index, (standard - present) ** 2) for index, standard in enumerate(standards)]
                distances.sort(key=lambda pair: pair[1])

                # compare to standard values and pick closest
                length = len(standards)
                closest = distances[0][0]
                triple = (closest - 1, closest, closest + 1)
                if closest == 0:
                    triple = (closest, closest + 1, closest + 2)
                if closest == length - 1:
                    triple = (length - 3, length - 2, length - 1)

                # find attempted values
                attempts = [standards[index] for index in triple]

                # meditate over standard values
                self.meditate(parameter, *attempts)

            # get improved values
            improvements = [self.__getattribute__(parameter) for parameter in parameters]

            # reassign
            status = improvements

            # add to recording
            recording.append(status)

        # invert recording
        tapes = [tape for tape in zip(*recording)]

        # monitor
        for parameter, value, tape in zip(parameters, status, tapes):
            print(parameter + ': ' + str(tape) + ', ' + str(value))

        return None

    def mull(self, inclusions, idea, distance=None, count=None):
        """Determine if removing a word shortens the distance from cluster centers in the claustrum spaces.

        Arguments:
            inclusions: list of strings, unmullable words
            idea: string, the idea in question
            distance=None: float, distance in claustrum space if previously determined
            count=None: integer, count of the number of mulling rounds

        Returns:
            (str, float) tuple, the idea with the smallest distance, this distance
        """

        # determine distance for unpruned idea:
        if distance is None:
            distance = self.reflect(idea)

        # initiate count
        if count is None:
            count = 0

        # adjust count
        count += 1

        # monitor
        if self.introspective:
            # print count
            pass

        # abort if count is expired
        if self.limit is not None:
            if count > self.limit:

                return idea, distance

        # if the length is only 1, return it
        if len(idea.split()) < 2:

            return idea, distance

        # begin scorecard with untrimmed idea
        scores = [(idea, distance)]

        # remove one word at a time and calculate the distance
        words = idea.split()
        length = len(words)
        for index, word in enumerate(words):

            # skip if word in inclusions
            if word in inclusions:
                continue

            # omit each word
            trial = words[:index] + words[index + 1:]
            trial = ' '.join(trial)

            # determine distance and append
            score = self.reflect(trial)
            scores.append((trial, score))

        # sort by score and pick the top
        scores.sort(key=lambda pair: pair[1])
        best = scores[0]

        # if a change was made, attempt to mull the new result
        if best != (idea, distance):

            return self.mull(inclusions, best[0], best[1], count)

        return best

    def muse(self, inclusions=None, notions=None):
        """Whittle down a long random string of words.

        Arguments:
            inclusions=Nones: list of strings, words to be included
            notions=None: diction mapping indices to words

        Returns:
            string, list of words
        """

        # set defaults
        if inclusions is None:
            inclusions = []
        if notions is None:
            notions = {}

        # establish target line length
        self.intend()

        # retrieve words and their frequencies, excluding inclusions and omissions
        omissions = self.omissions
        frequencies = self.hippocampus.frequencies.items()
        frequencies = [(word, frequency) for word, frequency in frequencies if word not in inclusions + omissions]

        # sort words randomly with a partial weighting for frequency adjusted by the structure parameter
        ranking = [(word, rand() + self.structure * frequency) for word, frequency in frequencies]
        ranking.sort(key=lambda pair: pair[1], reverse=True)

        # scrape off the top capacity's worth and add inclusions
        capacity = self.capacity - len(inclusions)
        words = [word for word, score in ranking[:capacity]]
        words += inclusions

        # randomize order
        words.sort(key=lambda word: rand())

        # monitor
        if self.introspective:
            #print ' '.join(words)
            pass

        # apply notions
        for word, position in notions.items():

            # switch words to satisfy notions
            occupier = words[position]
            hole = words.index(word)
            words[hole] = occupier
            words[position] = word

        # monitor
        if self.introspective:
            #print ' '.join(words)
            pass

        # pick random cluster centers
        #center = choice(self.claustrum.centers)
        #isocenter = choice(self.isoclaustrum.centers)

        # default centers to the first centers
        center = self.claustrum.centers[0]
        isocenter = self.isoclaustrum.centers[0]

        # find the closest claustrum cluster
        balance = self.balance
        if balance < 1.0:
            features = self.parse(words)
            casting = self.claustrum.cast(features)
            center = self.claustrum.site(casting)
            pass

        # find the closest isoclaustrum cluster
        if balance > -1.0:
            features = self.isoparse(words)
            casting = self.isoclaustrum.cast(features)
            isocenter = self.isoclaustrum.site(casting)
            pass

        # pick random samples from claustrum and isoclaustrum to use as target centers
        #sample = choice(self.claustrum)
        #center = sample['projection']
        #sample = choice(self.isoclaustrum)
        #isocenter = sample['projection']

        # time point
        if self.perceptive:
            print(' ')
            print('beginning new attempt:')
            self.mark()

        # begin whittling down
        ambition = self.ambition
        whimsy = self.whimsy
        while len(words) > ambition:

            # timepoint header
            if self.perceptive:
                print(' ')
                print('length %d:' % len(words))

            # resite centers after every few rounds
            if len(words) % 5 == 0:

                # find the closest claustrum cluster
                balance = self.balance
                if balance < 1.0:
                    features = self.parse(words)
                    casting = self.claustrum.cast(features)
                    center = self.claustrum.site(casting)
                    pass

                # find the closest isoclaustrum cluster
                if balance > -1.0:
                    features = self.isoparse(words)
                    casting = self.isoclaustrum.cast(features)
                    isocenter = self.isoclaustrum.site(casting)
                    pass

            # retrieve valid indices by adhering to notions and inclusions
            positions = [index for index, word in enumerate(words) if word not in [key for key in notions.keys()] + inclusions]

            # if diligence is set, only check a subset of positions
            if self.diligence is not None:

                # pick a random subset
                diligence = self.diligence
                positions.sort(key=lambda member: rand())
                positions = positions[:diligence]

            # get a distance score based on removing each word
            scores = []
            for position in positions:

                # construct a new sequence by removing the word
                removal = words[:position] + words[position + 1:]
                removal = ' '.join(removal)

                # measure the distance score
                score = self.reflect(removal, [center], [isocenter])
                scores.append((removal, score))

                # timepoint
                if self.perceptive:
                    self.mark()

            # sort scores and pick closest
            scores.sort(key=lambda pair: abs(pair[1] - whimsy))
            words = scores[0][0].split()

            # monitor
            if self.introspective:
                #measurement = scores[0][1]
                #print measurement, ' '.join(words)
                #self.posit(' '.join(words))
                pass

        # form idea
        idea = ' '.join(words)

        return idea

    def parse(self, words, target=None):
        """Parse a list of words into a feature space vector, scaled based on a target length.

        Arguments:
            words: list of strings, words in the lyric
            target=None: integer, scale the vector based on a target length

        Return:
            list of (string, float) tuples, the features and their positions
        """

        # set default target to current length
        length = len(words)
        if target is None:
            target = length

        # scale positions to target length
        scaling = float(target) / float(length)
        features = [(word, (index * scaling) + 1) for index, word in enumerate(words)]

        return features

    def perform(self, *givens):
        """Compose and sing a song.

        Arguments:
            *givens: unpacked tuple of integers and strings

        Returns:
            None
        """

        # set defaults at random
        length = choice([1, 2, 3, 4])
        scheme = choice(['abcb', 'aa', 'abb', 'aabba'])
        themes = [choice(self.hippocampus.vocabulary), choice(self.hippocampus.vocabulary)]

        # replace defaults with givens
        if len(givens) > 0:

            # try to find a number for the length, otherwise default to 1
            length = 1
            numbers = [given for given in givens if str(given) != given]
            if len(numbers) > 0:
                length = numbers[0]

            # get strings
            alphas = [given for given in givens if str(given) == given]

            # replace scheme with first string, but ignore if empty
            if len(alphas) > 0:
                first = alphas[0]
                if len(first) > 1:
                    if first != ' ':
                        scheme = first

            # replace themes with second strings and thereafter
            if len(alphas) > 1:

                # split all strings by spaces
                themes = alphas[1:]
                themes = [word for theme in themes for word in theme.split()]

        # compose and sing a song
        self.compose(length, scheme, *themes)
        self.sing()

        # record?
        if self.auto:
            self.record()

        return None

    def phrase(self, concept, notions=None):
        """Phrase a concept, finding the best word order by examining the closest cluster centers.

        Arguments:
            concept: string, words in an order
            notions: dictionary mapping words to imposed positions

        Returns:
            (string, float) tuple, words in the best order and the tension score
        """

        # deconstruct concept into words, setting each location equal to its index
        words = concept.split()
        size = len(words)
        indices = [index for index in range(size)]
        locations = indices

        # create default notions as empty dictionary
        if notions is None:
            notions = {}

        # projection concept into claustrum space
        frequencies = self.hippocampus.frequencies
        features = [(word, location + 1.0) for location, word in enumerate(words)]
        projection = self.claustrum.project(features)

        # measure squared distance to all claustrum cluster centers
        centers = self.claustrum.centers
        distances = [sum([(b - a) ** 2 for a, b in zip(projection, center)]) for center in centers]

        # keep the three closest centers for reconstructions
        distances = [(index, distance) for index, distance in enumerate(distances)]
        distances.sort(key=lambda pair: pair[1])
        closest = [centers[index] for index, distance in distances[:3]]
        reconstructions = [self.claustrum.reconstruct(point) for point in closest]

        # projection concept into isoclaustrum space
        length = float(len(words))
        features = [(word, location - length) for location, word in enumerate(words)]
        projection = self.isoclaustrum.project(features)

        # measure squared distance to all claustrum cluster centers
        centers = self.isoclaustrum.centers
        distances = [sum([(b - a) ** 2 for a, b in zip(projection, center)]) for center in centers]

        # keep the three closest centers for reconstructions
        distances = [(index, distance) for index, distance in enumerate(distances)]
        distances.sort(key=lambda pair: pair[1])
        closest = [centers[index] for index, distance in distances[:3]]
        reconstructions += [self.isoclaustrum.reconstruct(point) for point in closest]

        # transform each reconstruction into an arrangement
        arrangements = []
        for reconstruction in reconstructions:

            # prune to relevant words
            catalog = dict(reconstruction)
            reconstruction = [(word, catalog[word]) for word in words]

            # sort by position
            reconstruction.sort(key=lambda pair: pair[1])

            # pick off words to get an arrangement
            arrangement = [pair[0] for pair in reconstruction]
            arrangements.append(arrangement)

        # swap words in the arrangements in order to satisfy notions
        for arrangement in arrangements:

            # go through each notion and swap accordingly
            for word, location in notions.items():

                # determine swapped word and its location
                swap = arrangement[location]
                hole = arrangement.index(word)

                # perform switch
                arrangement[location] = word
                arrangement[hole] = swap

        # join arrangements
        arrangements = [' '.join(arrangement) for arrangement in arrangements]

        # determine claustrum scores
        scores = [(arrangement, self.reflect(arrangement)) for arrangement in arrangements]

        # sort for the best arrangement
        scores.sort(key=lambda pair: pair[1])

        # grab top
        idea = scores[0][0]

        return idea

    def ponder(self):
        """Ponder the lyrics, digesting the contextual relationships.

        Arguments:
            None

        Returns:
            None
        """

        # record the contextual relationships amongst the words
        self.hippocampus.digest()
        self.hippocampus.trim()

        # generate histogram of line lengths
        self.hippocampus.census()

        return None

    def posit(self, lyric):
        """Posit a lyric, marking it in the claustrum spaces.

        Arguments:
            lyric: string, a hypothetical lyric

        Returns:
            None
        """

        # deconstruct lyric
        words = lyric.split()
        length = float(len(words))

        # project into claustrum space
        label = '____{' + lyric + '}'
        features = [(word, index + 1.0) for index, word in enumerate(words)]
        self.claustrum.project(features, annotation=label)

        # project into isoclaustrum space
        features = [(word, index - length) for index, word in enumerate(words)]
        self.isoclaustrum.project(features, annotation=label)

        return None

    def recite(self, title):
        """Recite given song title.

        Arguments:
            title: string, song title

        Returns:
            None
        """

        # get list of songs and sort by closeness to given title
        songs = [song for artist in self for song in artist]
        songs.sort(key=lambda song: self.gauge(title, song.name), reverse=True)

        # view top song
        songs[0].view()

        return None

    def record(self, name=None, index=-1):
        """Record the composition into a file.

        Arguments:
            name=None: string, file name
            index=-1: integer, composition index

        Returns:
            None
        """

        # get song
        song = self.compositions[index]

        # condense the last line for the file name if not given
        if name is None:
            name = song[-1][-1]
            name = name.replace(' ', '')
            name = name[:20]
        name = 'Recordings/' + name + '.txt'

        # create a file with the name
        recording = open(name, 'w')

        # populate lines, verse by verse
        lines = []
        for verse in song:

            # line by line
            for line in verse:

                # add lines
                line += '\n'
                lines.append(line)

            # add blank for separation
            lines.append(' \n')

        # three more blank lines
        for count in range(3):
            lines.append(' \n')

        # list influences
        for artist in self:
            lines.append(artist.name + ' \n')

            # list songs
            for song in artist:

                # only list selected songs
                if song.selected:
                    lines.append('    ' + song.name + ' \n')

        # write to file
        recording.writelines(lines)

        # close file
        recording.close()

        return None

    def reflect(self, idea, centers=None, isocenters=None, target=None):
        """Reflect on an idea, determining it's closeness to clusters in the claustrum and isoclaustrum.

        Arguments:
            idea: string, line of lyrics
            centers=None: list of lists of floats, specific claustrum cluster centers to use for measuring
            isocenters=None: list of lists of floats, specific isoclaustrum cluster centers to use for measuring
            target: integer, target length of the idea

        Returns:
            float, distance score
        """

        # default centers to all centers
        if centers is None:
            centers = self.claustrum.centers

        # default isocenters to all isocenters
        if isocenters is None:
            isocenters = self.isoclaustrum.centers

        # split idea into words
        words = idea.split()
        length = len(words)

        # default target to length
        if target is None:
            target = length

        # project into claustrum space
        balance = self.balance
        distance = 0.0
        if balance < 1.0:
            features = self.parse(words, target)
            projection = self.claustrum.cast(features)

            # calculate the minimum distance to a cluster center
            distances = [sum([(b - a) ** 2 for a, b in zip(center, projection)]) for center in centers]
            distance = min(distances)

        # project into isoclaustrum space
        isodistance = 0.0
        if balance > -1.0:
            features = self.isoparse(words, target)
            projection = self.isoclaustrum.cast(features)

            # calculate the minimum distance to a cluster center and add to previous minimum
            distances = [sum([(b - a) ** 2 for a, b in zip(center, projection)]) for center in isocenters]
            isodistance = min(distances)

        # determine claustrum, isoclaustrum balance from equalizer
        if balance < 0:
            left = -balance * 0.5 + 0.5
            right = 1.0 - left
        else:
            right = balance * 0.5 + 0.5
            left = 1.0 - right

        # calculate score
        score = left * distance + right * isodistance

        return score

    @staticmethod
    def relate(wordi, wordii):
        """Relate a word to another word, based on common letters in common beginning positions.

        Arguments:
            wordi: string, the first word
            wordii: string, the second word

        Returns:
            float, comparison score
        """

        # create feature vector for each word
        vectors = []
        for word in [wordi, wordii]:

            # establish vector
            vector = {}

            # go through each letter
            for position, letter in enumerate(word):

                # set vector entry as the reciprocal of the position (plus 1)
                # unless there is already an entry
                if letter not in vector.keys():
                    vector[letter] = 1 / float(position + 1)

            # append vector
            vectors.append(vector)

        # create list of letters
        letters = [key for key in vectors[0].keys()] + [key for key in vectors[1].keys()]
        letters = list(set(letters))

        # calculate overall score
        score = 0.0
        for letter in letters:

            # add the square of each letter's difference
            distance = vectors[1].setdefault(letter, 0.0) - vectors[0].setdefault(letter, 0.0)
            score += distance ** 2

        return score

    def remember(self, target):
        """Remember where a word was found.

        Arguments:
            query: string

        Returns:
            None
        """

        # check for presence in themes
        presence = [target in song for song in self.hippocampus.sequences]

        # zip to song artist and titles
        songs = [(artist, song) for artist in self for song in artist if song.selected]
        queries = zip(songs, presence)
        answers = [item for item, answer in queries if answer is True]

        # print to screen
        print(' ')
        for artist, song in answers:
            print(artist.name, song.name)

            # find the particular lines and print them
            for line in song:

                # break into words and lowercase
                words = line.split()
                words = [word.lower() for word in words]

                # strip off punctuation
                for symbol in [',', '!', ';', '..', '...', '-']:
                    words = [word.strip(symbol) for word in words]

                if target in words:
                    print('    ' + line)

            # print spacer
            print(' ')

        return None

    def rhyme(self, wordi, wordii):
        """Calculate a rhyme score by relating backwards versions of words.

        Arguments:
            wordi: string, first word
            wordii: string, second word

        Returns:
            float, rhyming score
        """

        # flip word one
        wordi = [letter for letter in wordi]
        wordi.reverse()
        wordi = ''.join(wordi)

        # flip word two
        wordii = [letter for letter in wordii]
        wordii.reverse()
        wordii = ''.join(wordii)

        # relate for score
        score = self.relate(wordi, wordii)

        return score

    def seek(self, letter=None, entries=3):
        """Seek the artist spelling in the first entry az lyrics pages.

        Arguments:
            letter: string, the letter in the index to search, defaults to first letter
            entries=3: integer, number of closest matches given

        Returns:
            None
        """

        # pass call to the first artist
        self[0].seek(letter, entries)

        return None

    def sing(self, index=-1):
        """Sing a song composed by the Bard.

        Arguments:
            index=-1: index of song to be song, the last composed by default

        Returns:
            None
        """

        # retrieve song
        song = self.compositions[index]

        # print each verse
        for verse in song:

            # print each lyric
            for lyric in verse:
                print(lyric)

            # print spacer
            print(' ')

        return None

    def study(self, dimensions=None, resolution=None):
        """Study to prepare to sing songs.

        Arguments:
            dimensions=None: integer, number of dimensions in the pca deconstructions
            resolution=None: integer, number of cluster in the kmeans analysis

        Returns:
            None
        """

        # set default dimensions
        if dimensions is None:
            dimensions = self.dimensions

        # set default resolution
        if resolution is None:
            resolution = self.resolution

        # number of selected songs
        songs = [song for artist in self for song in artist if song.selected]
        number = len(songs)
        print('songs: %d' % number)

        # prepare for singing
        print('listening...')
        self.listen()
        print('pondering...')
        self.ponder()
        #self.contextualize(dimensions, resolution)
        #self.categorize(resolution)
        #self.generalize()
        #self.conceptualize(dimensions, resolution)
        self.synergize(dimensions, resolution)

        return None

    def synergize(self, dimensions=None, resolution=None):
        """Develop the claustrum and isocalustrum attributes to model each lyric as a point in utterance space.

        Arguments:
            dimensions=None: integer, number of pca component axes
            resolution=None: integer, number of concept clusters

        Returns:
            None
        """

        # set default dimensions
        if dimensions is None:
            dimensions = self.dimensions

        # set default resolution
        if resolution is None:
            resolution = self.resolution

        # construct frequency weightings
        frequencies = self.hippocampus.frequencies
        if not self.skew:
            frequencies = {word: 1.0 for word, frequency in frequencies.items()}

        # create a data point for each lyric where a feature is the position (index + 1) of each word
        data = []
        for lyric in self.hippocampus.lyrics:
            words = lyric.split()
            features = self.parse(words)
            data.append((lyric, features))

        # create claustrum instance
        self.claustrum = Grammar(data, frequencies)

        # perform machine learning analysis
        self.claustrum.decompose(dimensions)
        self.claustrum.cluster(resolution)

        # prune off outlying clusters and recluster
        if self.outliers is not None:

            # change negative indices to positive indices
            prunes = []
            for outlier in self.outliers:
                member = outlier
                if outlier < 0:
                    member = resolution + outlier
                prunes.append(member)

            # perform pruning
            self.claustrum.prune(*prunes)
            self.claustrum.decompose(dimensions)
            self.claustrum.cluster(resolution)

        # create a data point for each lyric where a feature is the negative index for each word
        data = []
        for lyric in self.hippocampus.lyrics:
            words = lyric.split()
            features = self.isoparse(words)
            data.append((lyric, features))

        # create neoclaustrum instance
        self.isoclaustrum = Grammar(data, frequencies)

        # perform machine learning analysis
        self.isoclaustrum.decompose(dimensions)
        self.isoclaustrum.cluster(resolution)

        # prune off outlying clusters and recluster
        if self.outliers is not None:
            self.isoclaustrum.prune(*prunes)
            self.isoclaustrum.decompose(dimensions)
            self.isoclaustrum.cluster(resolution)

        return None

    def suppose(self, *inclusions):
        """Generate a line template from a point in Wernicke's space.

        Arguments:
            *inclusions: unpacked tuple of words to be included

        Returns:
            None
        """

        # pick a point from a random sampling and reconstruct the feature intensities
        point = self.wernickes.pick()
        reconstruction = self.wernickes.reconstruct(point)

        # monitor
        if self.introspective:
            #print point
            #print reconstruction
            pass

        # sort by category number
        reconstruction.sort(key=lambda feature: feature[0])

        # find categories of included words and subtract from reconstruction
        for word in inclusions:
            number = self.cortex.registry[word]
            category = self.cortex[number]['cluster']
            intensity = reconstruction[category][1] - 1
            reconstruction[category] = (category, intensity)

        # query the histogram for a line length
        length = 2
        rando = rand()
        histogram = [bar for bar in self.hippocampus.histogram]
        histogram = histogram[2:-6]
        histogram.reverse()
        for category, weight in histogram:
            if rando < weight:
                length = category

        # subtract by number of inclusions
        length -= len(inclusions)

        # reform reconstruction into list of cluster instances
        instances = []
        for round in range(length):

            # sort reconstructions by weight and append highest category
            reconstruction.sort(key=lambda feature: feature[1], reverse=True)
            category = reconstruction[0][0]
            weight = reconstruction[0][1]
            instances.append(category)
            reconstruction[0] = (category, weight - 1.0)

        # monitor
        if self.introspective:
            #print instances
            pass

        # set template
        template = instances + list(inclusions)
        self.template = template

        return None

    def view(self):
        """View the song repertoire of the bard.

        Arguments:
            None

        Returns:
            None
        """

        # print each artist
        for number, artist in enumerate(self):
            print(str(number) + ') ' + str(artist))

            # print each song
            for index, song in enumerate(artist):

                # make string depending on status
                indicator = '   '
                if song.selected:
                    indicator = ' X '
                print(indicator + str(index) + ': ' + str(song))

            print(' ')

        return None


# Execution
def check(bard, *samples):

    for sample in samples:
        lyric = bard.claustrum[sample]['name']
        words = lyric.split()
        features = [(word, index + 1.0) for index, word in enumerate(words)]

        projection = bard.claustrum.project(features)
        casting = bard.claustrum.cast(features)
        bard.claustrum.reconstruct(projection, annotation = '(' + str(sample) + ')')
        bard.claustrum.reconstruct(casting, annotation = '(' + str(sample) + ')*')

    return None

#z = Bard('ledzeppelin', 'jimihendrix', 'deeppurple', 'faint', 'archenemy', 'muse', 'pinkfloyd', 'blacksabbath', 'nirvana', 'aliceinchains', 'avengedsevenfold', 'blueoystercult', 'bush', 'bjork')
#z = Bard('cake', 'cranberries', 'elvis', 'falloutboy', 'cure', 'dio', 'eminem', 'goldenearring', 'inxs', 'jarsofclay', 'blueoystercult')
#z = Bard('archenemy', 'dio', 'blueoystercult', 'wutang', 'faint', 'bjork', 'blindmelon', 'chuckberry', 'cream', 'europe')
#z = Bard('bush', 'jimihendrix', 'cure', 'difranco', 'cake', 'cagetheelephant', 'avengedsevenfold', 'depeche', 'doors', 'dylan', 'blacksabbath', 'amos', 'adele', 'aliceinchains', 'guns', 'killers', 'gorillaz', 'ghostbc', 'gabriel', 'yardbirds', 'west')
z = Bard('ledzeppelin', 'pinkfloyd', 'tool')
#z = Bard('bush', 'jimihendrix', 'cure', 'difranco', 'cake', 'cagetheelephant', 'avengedsevenfold', 'depeche', 'doors', 'dylan', 'blacksabbath', 'amos', 'adele', 'aliceinchains', 'guns', 'killers', 'gorillaz', 'ghostbc', 'gabriel', 'yardbirds', 'west', 'archenemy', 'dio', 'blueoystercult', 'chuckberry', 'ledzeppelin', 'faint', 'blindmelon', 'cranberries', 'falloutboy', 'goldenearring', 'bjork', 'jarsofclay', 'nirvana', 'deeppurple', 'pinkfloyd', 'tool', 'unleashthearchers', 'swans')
z.include()
z.study()

