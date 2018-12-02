import os
import time
import tensorflow as tf
from functools import wraps
from inspect import getargspec
from pathlib import Path
from azureml.core.run import Run

# pylint: disable-msg=E0611
from tensorflow.python.tools import freeze_graph as freeze
# pylint: enable-msg=E0611

def info(msg, char = "#", width = 75):
    print("")
    print(char * width)
    print(char + "   %0*s" % ((-1*width)+5, msg) + char)
    print(char * width)

def print_info(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        info('-> {}'.format(f.__name__))
        print('Parameters:')
        ps = list(zip(getargspec(f).args, args))
        width = max(len(x[0]) for x in ps) + 1
        for t in ps:
            items = str(t[1]).split('\n')
            print('   {0:<{w}} ->  {1}'.format(t[0], items[0], w=width))
            for i in range(len(items) - 1):
                print('   {0:<{w}}       {1}'.format(' ', items[i+1], w=width))
        ts = time.time()
        result = f(*args, **kwargs)
        te = time.time()
        print('\n -- Elapsed {0:.4f}s\n'.format(te-ts))
        return result
    return wrapper

def check_dir(path, check=False):
    if check:
        assert os.path.exists(path), '{} does not exist!'.format(path)
    else:
        if not os.path.exists(path):
            os.makedirs(path)
        return Path(path).resolve()

def aml_log(run, **kwargs):
    if run != None:
        for key, value in kwargs.items():
            run.log(key, value)
    else:
        print('{}'.format(FormatDict(kwargs)))

class FormatDict:
    def __init__(self, dictionary):
        self.dictionary = dictionary
        self.max_key = max([len(l) for l in self.dictionary.keys()])
        self.max_val = max([len(str(l)) for l in self.dictionary.values()]) 

    def __format__(self, fmt):
        l = list(self.dictionary.items())
        s = []
        s.append('-'*(self.max_key + self.max_val + 7) + '\n' )
        for item in l:
            s.append('| {k:<{kw}} | {v:<{vw}} |\n'.format(k=str(item[0]), 
                                        kw=self.max_key, 
                                        v=str(item[1]), 
                                        vw=self.max_val))
        s.append('-'*(self.max_key + self.max_val + 7) + '\n' )
        return ''.join(s)

def save_model(sess, export_path, output_node):
    # saving model
    checkpoint = str(export_path.joinpath('model.ckpt').resolve())

    saver = tf.train.Saver()
    saver.save(sess, checkpoint)

    # graph
    tf.train.write_graph(sess.graph_def, str(export_path), "model.pb", as_text=False)

    # freeze
    g = os.path.join(export_path, "model.pb")
    frozen = os.path.join(export_path, "digits.pb")

    freeze.freeze_graph(
        input_graph = g, 
        input_saver = "", 
        input_binary = True, 
        input_checkpoint = checkpoint, 
        output_node_names = output_node,
        restore_op_name = "",
        filename_tensor_name = "",
        output_graph = frozen,
        clear_devices = True,
        initializer_nodes = "")

    print("Model saved to " + frozen)
