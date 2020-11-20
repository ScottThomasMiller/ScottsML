'''
This code uses GA to do a search for the hyperparameters of PyTorch model.

- number of hidden layers
- number of neurons (depth)
- optimizer
- cost function

'''

import os

import datetime
import math
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import timeit
import torch
from utils import *

def logprint(msg):
  global logfile
  print_str = "["+str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))+"] "+str(msg)
  print(print_str)
  if logfile is not None:
    logfile.write(print_str+'\n')
    logfile.flush()
    os.fsync(logfile.fileno())

class hypertorch:
  ''' hypertorch contains a genetic algorithm (GA) which searches for the best combination of hyperparameters of any
      PyTorch module. '''
  def __init__(self, X_dev, Y_dev, X_test, Y_test, 
               cache_filename="hypertorch_cache.npy", genmax = 15, gennum = 25, ncontestants = 5, mutpct = 0.4, mutscale = 0.5):
    self.X_dev = X_dev 
    self.Y_dev = Y_dev
    self.X_test = X_test 
    self.Y_test = Y_test
    self.num_features = X_dev.shape[1] 
    self.num_classes = Y_dev.shape[1]
    self.load_cache(cache_filename)
    self.genmax = genmax
    self.gennum = gennum
    self.ncontestants = ncontestants
    self.mutpct = mutpct
    self.mutscale = mutscale

    self.cache_size = 0
    self.num_cached = 0
    self.cache = None
    self.cache_filename = '/content/gdrive/My Drive/hypertorch/' + cache_filename

  init_cache = 999
  nchrom = 3  ''' number of chromosomes/hyperparameters '''

  ''' The following are lookup arrays for all chromosome values.  The GA populations
      will be arrays of nchrom-length vectors.  Each position in each vector is an
      index into one of the following lists: '''
  batch_sizes = [2,4,16,32,64,128,256]
  learn_rates = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
  opt_funcs = [
    'adadelta',
    'adagrad',
    'adam',
    'adamax',
    'adamw',
    'asgd',
    'rmsprop',
    'rprop',
    'sgd',
    'sparseadam']

  ''' perhaps future code will be able to alter an instantiated PyTorch module and change these: '''
  #cost_funcs = [  
  #  'mean_squared_error',
  #  'mean_squared_logarithmic_error',
  #  'categorical_hinge',
  #  'categorical_crossentropy']
  #nodes = np.linspace(2,10,5, dtype=np.int32)*100
  #layers = np.linspace(1,10,10, dtype=np.int32)
  #act_funcs = [
  #    'relu',
  #    'sigmoid',
  #    'tanh'
  #]

  def load_cache(self, cache_filename):
    try:
      if self.cache_filename:
        logprint("loading cache file "+self.cache_filename)
        self.cache = np.load(self.cache_filename)      
    except:
      if not self.cache:
        logprint("initializing new cache.")
        self.cache = hypertorch.init_cache * np.ones((len(self.cost_funcs), len(self.opt_funcs), len(self.learn_rates)))
                                                #len(self.nodes), len(self.layers), len(self.act_funcs)))
    finally:
        self.cache_size = len(self.cache[self.cache >= 0])
        self.num_cached = len(self.cache[self.cache < hypertorch.init_cache])
        logprint("total cache size: "+str(self.cache_size))
        logprint("num cached: "+str(self.num_cached))
  
  # run_dev creates and runs the keras model on the dev set using the given chromosome:
  def run_dev(self, Z):
    cost_func = hypertorch.cost_funcs[Z[0]]
    opt_func = hypertorch.opt_funcs[Z[1]]
    learn_rate = hypertorch.learn_rates[Z[2]]

    my_model = Sequential()
    my_model.add(Dense(num_nodes, input_dim=self.num_features, activation=act_func))
    for n in range(num_layers):
      my_model.add(Dense(num_nodes, activation=act_func))
    my_model.add(Dense(self.num_classes, activation='softmax'))
    
    tpu_model = tf.contrib.tpu.keras_to_tpu_model(
      my_model,
      strategy=tf.contrib.tpu.TPUDistributionStrategy(
        tf.contrib.cluster_resolver.TPUClusterResolver(tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])
      )
    )

    tpu_model.compile(
      optimizer=opt_func,
      loss=cost_func,
      metrics=['accuracy']
    )

    def train_gen(batch_size):
      while True:
        offset = np.random.randint(0, self.X_dev.shape[0] - batch_size)
        yield self.X_dev[offset:offset+batch_size], self.Y_dev[offset:offset + batch_size]

    try:
      tpu_model.fit_generator(
        train_gen(1024),
        epochs=10,
        steps_per_epoch=100,
        validation_data=(self.X_test, self.Y_test),
        verbose=0
      )
    except Exception as e:
      logprint("Exception:")
      logprint(str(e))
      self.show_z(Z)
    finally:
      return tpu_model
  
  def show_z(self, Z):
    ''' Debug-print the given chromosome. '''
    print("cost func: ", self.cost_funcs[Z[0]])
    print("opt func: ", str(self.opt_funcs[Z[1]]))
    print("learn rate: ", self.learn_rates[Z[2]])
    
  def train(self, W):
    ''' Train is called by the main GA loop to compute the test error for each
    ''' candidate parent in W, based on models derived from the dev set. '''
    nerr=np.zeros(len(W))
    i = 0
    for Z in W:
      logprint("training: "+str(Z))
      cache_val = self.cache[tuple(Z)]
      if cache_val < hypertorch.init_cache:
        nerr[i] = cache_val
      else:
        tpu_model = self.run_dev(Z)
        cpu_model = tpu_model.sync_to_cpu()
        scores = cpu_model.evaluate(self.X_test, self.Y_test, verbose=0)
        nerr[i] = (100-scores[1]*100)
        self.cache[tuple(Z)] = nerr[i]
      i += 1
    return nerr
  
  def crossbreed(self, parent1, parent2):      
    ''' Return the child of both parents, taking each gene from only one parent at random. '''
    child = np.random.rand(hypertorch.nchrom)
    for n in range(hypertorch.nchrom):
      rando = np.random.rand(1)
      child[n] = rando*parent1[n] + (1-rando)*parent2[n]
    return(child)

  def dmutate(self, Muteys, delta, cap_value):    
    ''' Translate a continuous mutation delta into an integer hyperparameter index. '''
    New = Muteys.astype(np.float64) + delta
    Muteys[New >= 0] = np.rint(New[New >= 0]) % cap_value
    Muteys[New < 0] = cap_value - 1;
    return (Muteys)
    
  def mutate(self, newpop):
    ''' Return the new, mutated population. '''
    num_of_mutation=math.ceil(len(newpop)*self.mutpct)
    mutei=np.random.choice(len(newpop),num_of_mutation, replace=False, p=None)
    delta = np.random.normal(0,self.mutscale)
    newpop[mutei,0] = self.dmutate(newpop[mutei,0], delta, len(self.cost_funcs))
    newpop[mutei,1] = self.dmutate(newpop[mutei,1], delta, len(self.opt_funcs))
    newpop[mutei,2] = self.dmutate(newpop[mutei,2], delta, len(self.learn_rates))
    return(newpop)

  def fresh_pop():
    ''' Return the initial population. '''
    logprint("building new population")
    # is there a numpy way to do this instead of looping?
    pop_size = len(hypertorch.cost_funcs) * len(hypertorch.opt_funcs) * len(hypertorch.learn_rates)
               #len(hypertorch.nodes) * len(hypertorch.layers) * len(hypertorch.act_funcs)
    newPop = np.zeros((pop_size,hypertorch.nchrom), dtype=np.int32)
    i = 0                                                                                                 
    for ibsize in range(len(hypertorch.batch_sizes)):
      for iopt in range(len(hypertorch.opt_funcs)):
        for ilr in range(len(hypertorch.learn_rates)):
          Z = np.array([ibsize, iopt, ilr], dtype=np.int32)
          newPop[i,:] = Z
          i += 1                                                                                         
    return newPop    
    
  def best_h(self):
    ''' The central algorithm, which mutates the population, searching for the best combo of genes. '''
    oldpop = hypertorch.fresh_pop()
    newpop = population.copy()
    popsize = len(population)
    logprint(f"best_h->population size: {popsize}, total epochs: {self.genmax*self.gennum}")
    
    # if the cache is full, then there is no point to running the GA:
    if (self.cache_size > 0) & (self.cache_size == self.num_cached):
      logprint("cache is full.  short-circuit.")
      return (np.unravel_index(np.argmin(self.cache, axis=None), self.cache.shape))
    
    for i in range(self.genmax):       
      for j in range(self.gennum):  
        candidates=oldpop[np.random.choice(len(oldpop), size=self.ncontestants, replace=False)] 
        parent1=candidates[np.argmin(self.train(candidates)),:]
        candidates=oldpop[np.random.choice(len(oldpop), size=self.ncontestants, replace=False)] 
        parent2=candidates[np.argmin(self.train(candidates)),:]
        newpop[j,:]=self.crossbreed(parent1,parent2)
        
        argmin_cache = np.unravel_index(np.argmin(self.cache, axis=None), self.cache.shape)
        np.save(self.cache_filename, self.cache)
        logprint("gen: "+str(i)+", pop: "+str(j)+", min cache: "+str(self.cache[argmin_cache])+", argmin cache: "+str(argmin_cache))
      newpop=self.mutate(newpop)  
      oldpop=newpop.copy()
      
    H = np.unravel_index(np.argmin(self.cache, axis=None), self.cache.shape)
    return(H)

