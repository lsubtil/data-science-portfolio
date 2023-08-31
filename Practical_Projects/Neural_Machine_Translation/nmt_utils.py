import random
import numpy as np
import matplotlib.pyplot as plt

from faker import Faker
from tqdm import tqdm
from babel.dates import format_date

from tensorflow.keras.utils import to_categorical
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model


fake = Faker()
Faker.seed(12345)
random.seed(12345)

# Define format of the data we would like to generate
formats = ['short',
           'medium',
           'long',
           'full',
           'full',
           'full',
           'full',
           'full',
           'full',
           'full',
           'full',
           'full',
           'full',
           'd MMM YYY', 
           'd MMMM YYY',
           'dd MMM YYY',
           'd MMM, YYY',
           'd MMMM, YYY',
           'dd, MMM YYY',
           'd MM YY',
           'd MMMM YYY',
           'MMMM d YYY',
           'MMMM d, YYY',
           'dd.MM.YY']

# change this if you want it to work with another language
locals = ['en_US']

def load_date():
    """
    Loads some fake dates 
    
    Returns:
        tuple containing human readable string, machine readable string, and date object
        
    """
    
    dt = fake.date_object()

    try:
        human_readable = format_date(dt, format = random.choice(formats), locale = random.choice(locals))
        human_readable = human_readable.lower()
        human_readable = human_readable.replace(',','')
        machine_readable = dt.isoformat()
        
    except AttributeError as e:
        
        return None, None, None

    return human_readable, machine_readable, dt


def load_dataset(m):
    """
    Loads a dataset with m examples and vocabularies
    
    Arguments:
        m -- the number of examples to generate
    
    Returns:
        dataset -- list; pairs of correspondent human and machine readable dates
        human -- dict; human vocabulary to index 
        machine -- dict; machine vocabulary to index
        inv_machine -- dict; index to machine vocabulary
    """
    
    human_vocab = set()
    machine_vocab = set()
    dataset = []
    

    for i in tqdm(range(m)): 
        h, m, _ = load_date()
        
        if h is not None:
            dataset.append((h, m))
            
            # define human vocabulary
            human_vocab.update(tuple(h))
            
            # define machine vocabulary
            machine_vocab.update(tuple(m))    
    
    # human_vocab to index
    human = dict(zip(sorted(human_vocab) + ['<unk>', '<pad>'], list(range(len(human_vocab) + 2))))    
    
    # machine_vocab to index
    inv_machine = dict(enumerate(sorted(machine_vocab)))
   
    # index to machine_vocab
    machine = {v:k for k,v in inv_machine.items()}    
 
    return dataset, human, machine, inv_machine


def preprocess_data(dataset, human_vocab, machine_vocab, Tx, Ty):
    
    X, Y = zip(*dataset)
    
    X = [string_to_int(i, Tx, human_vocab) for i in X]
    Y = [string_to_int(t, Ty, machine_vocab) for t in Y]
    
    Xoh = list(map(lambda x: to_categorical(x, num_classes = len(human_vocab)), X))
    Yoh = list(map(lambda x: to_categorical(x, num_classes = len(machine_vocab)), Y))

    return np.array(X), np.array(Y), np.array(Xoh), np.array(Yoh)


def string_to_int(string, length, vocab):
    """
    Converts a string in the vocabulary into a list of integers representing the positions of the string characters in the "vocab"
    
    Arguments:
        string -- input string, e.g. 'Wed 10 Jul 2007'
        length -- the number of time steps you'd like, determines if the output will be padded or cut
        vocab -- vocabulary, dictionary used to index every character of your "string"
    
    Returns:
        rep -- list of integers (or '<unk>') representing the position of the string's character in the vocabulary
    """
    
    string = string.lower()
    string = string.replace(',','')
    
    if len(string) > length:
        string = string[:length]
        
    rep = list(map(lambda x: vocab.get(x, '<unk>'), string))
    
    if len(string) < length:
        rep += [vocab['<pad>']] * (length - len(string))

    return rep


def int_to_string(ints, inv_vocab):
    """
    Output a machine readable list of characters based on a list of indexes in the machine's vocabulary
    
    Arguments:
        ints -- list of integers representing indexes in the machine's vocabulary
        inv_vocab -- dictionary mapping machine readable indexes to machine readable characters 
    
    Returns:
        l -- list of characters corresponding to the indexes of ints thanks to the inv_vocab mapping
    """
    
    l = [inv_vocab[i] for i in ints]
    return l


def softmax(x, axis=1):
    """
    Softmax activation function.
    
    Arguments:
        x -- Tensor
        axis -- Integer, axis along which the softmax normalization is applied
    
    Returns:
        Tensor, output of softmax transformation
        
    Raises:
        ValueError -- In case `dim(x) == 1`.
    """
    
    ndim = K.ndim(x)
    if ndim == 2:
        return K.softmax(x)
    
    elif ndim > 2:
        e = K.exp(x - K.max(x, axis=axis, keepdims=True))
        s = K.sum(e, axis=axis, keepdims=True)
        return e/s
    
    else:
        raise ValueError('Cannot apply softmax to a tensor that is 1D')