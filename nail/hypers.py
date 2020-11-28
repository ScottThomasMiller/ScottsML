'''
This code uses GA to do a search for the hyperparameters of PyTorch model.

- number of hidden layers
- number of neurons (depth)
- optimizer
- cost function

'''


import datetime
from glob import glob
import math
import numpy as np
import os
import re
from   sklearn.decomposition import PCA
from   sklearn.model_selection import train_test_split
import time
import timeit
import torch
from   nail.hnn.utils import *

tempdir  = 'Hypers.temp'
max_time = 15 # max runtime minutes for each job

def err_exit(msg):
  logmsg(f'ERROR: {msg}')
  exit(-1)

class Hypers():
  ''' hypers contains a genetic algorithm (GA) which searches for the best combination of hyperparameters of any
      PyTorch module. '''
  def __init__(self, 
               cache_filename="hypers_cache.npy", genmax = 15, gennum = 25, ncontestants = 5, mutpct = 0.4, mutscale = 0.5):
    self.genmax = genmax
    self.gennum = gennum
    self.ncontestants = ncontestants
    self.mutpct = mutpct
    self.mutscale = mutscale

    self.cache_size = 0
    self.num_cached = 0
    self.cache = None
    self.cache_filename = f'{cache_filename}'
    #self.cache_filename = f'/content/gdrive/My Drive/hypers/{cache_filename}'
    self.load_cache(cache_filename)

  init_cache = 999
  nchrom = 3  #number of chromosomes/hyperparameters 

  ''' The following is the lookup array for all chromosome values.  The GA populations
      will be arrays of nchrom-length vectors.  Each position in each vector is an
      index into one of the following lists: '''
  Hparams = {
    'batch_size' : ['2','4','16','32','64','128','256'],
    'learn_rate' : ['1e-5',' 1e-4',' 1e-3',' 1e-2',' 1e-1'],
    'optim' : [
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
      }

  ''' future hyperparameters to consider: 
  'cost_func' : [  
    'mean_squared_error',
    'mean_squared_logarithmic_error',
    'categorical_hinge',
    'categorical_crossentropy'],
  'nodes' : np.linspace(2,10,5, dtype=np.int32)*100,
  'layer' : np.linspace(1,10,10, dtype=np.int32),
  'act_func' : [
      'relu',
      'sigmoid',
      'tanh'
  ]
  '''

  def load_cache(self, cache_filename):
    try:
      logmsg(f"loading cache file {self.cache_filename}")
      self.cache = np.load(self.cache_filename)      
    except:
      logmsg("load failed. initializing new cache.")
      self.cache = Hypers.init_cache * np.ones((len(Hypers.Hparams['batch_size']), len(Hypers.Hparams['optim']), len(Hypers.Hparams['learn_rate'])))
    finally:
        self.cache_size = len(self.cache[self.cache >= 0])
        self.num_cached = len(self.cache[self.cache < Hypers.init_cache])
        logmsg(f"total cache size: {self.cache_size}")
        logmsg(f"num cached: {self.num_cached}")
        logmsg(f"cache shape: {self.cache.shape}")
  
  def show_z(self, Z):
    ''' Debug-print the given chromosome. '''
    logmsg("batch size: ", Hypers.Hparams['batch_size'][Z[0]])
    logmsg("opt func: ", Hypers.Hparams['optim'][Z[1]])
    logmsg("learn rate: ", Hypers.Hparams['learn_rate'][Z[2]])
    
  def running_jobs(self):
    ''' Return the number of currently running jobs.'''
    try:
      stream = os.popen('bjobs | grep -iv "no unfinished" | wc -l 2> /dev/null', 
                        stdout=subprocess.PIPE, stderr=subpress.PIPE)
    except:
      err_exit('cannot execute bjobs')
    stream.wait() 
    if stream.rescode > 0:
      err_exit(stream.stderr.read())
    rstr = stream.stdout.read()
    logmsg(f'running_jobs->rstr: {rstr}')
    return int(rstr)
 
  def cleanup(self):
    ''' Archive then remove output files from prior jobs. '''
    global tempdir
    tstamp  = str(datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
    os.system(f'mkdir -p {tempdir}')
    status = True if os.path.isdir(tempdir) else err_exit(f'invalid tempdir "{tempdir}"')
    zname = f'snapshot_{tstamp}.zip'
    if len(glob(f'{tempdir}/*')) > 0:
      logmsg(f'archiving tempdir {tempdir} into zip file {zname}')
      status = True if os.system(f'zip {zname} {tempdir}/*') == 0 else err_exit('cannot zip snapshot files')
      status = True if os.system(f'rm {tempdir}/*') == 0 else err_exit('cannot rm temp files')
    return status

  def jobswait(self):
    ''' Wait until there are no more running jobs, and then return. '''
    njobs = self.running_jobs()
    while njobs > 0:
      logmsg(f'waiting for jobs to complete: {njobs} running')
      time.sleep(3)
      njobs = self.running_jobs()
    return 0   

  def prep_job(self, Z, gid, cid):
    ''' Write a single .job file. '''
    global tempdir
    global max_time
    zstr = ''
    for i in range(len(Z)):
      zstr += f'{zstr}{str(Z[i])}'
    jname = f'{tempdir}/hypers_group{gid}_Z{zstr}.job'
    oname = f'{tempdir}/hypers_group{gid}_Z{zstr}.out'
    ename = f'{tempdir}/hypers_group{gid}_Z{zstr}.err'
    cwd = os.getcwd()
    # Z = np.array([ibsize, iopt, ilr], dtype=np.int32)
    bsize = Hypers.Hparams['batch_size'][Z[0]]
    optf = Hypers.Hparams['optim'][Z[1]]
    lr = Hypers.Hparams['learn_rate'][Z[2]]
    status = 0
    
    with open(jname, 'w') as ofile:
      ofile.writelines(f'#!/bin/tcsh\n')
      ofile.writelines(f'#BSUB -n 12\n')
      ofile.writelines(f'#BSUB -W {max_time}\n')
      ofile.writelines(f'#BSUB -R span[hosts=1]\n')
      ofile.writelines(f'#BSUB -o {oname}\n')
      ofile.writelines(f'#BSUB -e {ename}\n')
      ofile.writelines(f'{cwd}/hypers.sh --learn_rate {lr} --optim: {optf} --batch_size {bsize} --group {gid} --contestant {cid}\n')

    return 0

  def prep_jobs(self, groups):
    ''' Prepare the .job parameter files for all training jobs. '''
    total = 0
    for group_id in range(len(groups)):
      W = groups[group_id]
      contestant_id = 0
      for Z in W:
        cache_val = self.cache[tuple(Z)]
        if cache_val >= Hypers.init_cache: # no cache hit
          self.prep_job(Z, group_id, contestant_id)
          total += 1
          contestant_id += 1

    return total
    
  def update_cache(self, groups):
    ''' Parse the output files from the recently completed jobs, and collect the loss values from each job.
        Update the cache with the results. Return the number of files parsed.
        oname = f'{tempdir}/hypers_group{group_id}_Z{Z}.out' '''
    global tempdir
    results = np.zeros(groups.shape)
    cpatt = re.compile('CID: ([0-9\.]+)')
    gpatt = re.compile('GID: ([0-9\.]+)')
    lpatt = re.compile('LOSS: ([0-9\.]+)')
    nfiles = 0
    for fname in glob(f'{tempdir}/hypers_group*_Z*.out'):
      nfiles += 1
      with open(fname,'r') as infile:
        for line in infile.readlines():
          cmatch = cpatt.match(line)
          gmatch = gpatt.match(line)
          lmatch = lpatt.match(line)
          if cmatch is not None:
            contestant = int(cmatch.group(1)) 
          if gmatch is not None:
            group = int(gmatch.group(1)) 
          if lmatch is not None:
            loss = float(lmatch.group(1)) 
        Z = groups[group][contestant]
        self.cache[Z] = loss

    return nfiles

  def select_cache(self, groups):
    ''' Return the cache values for the groups, split into two groups, one for 
        each tournament. '''
    cand1, cand2 = (np.zeros((groups[0].shape[0],)), np.zeros((groups[0].shape[0],)))
    for c in range(groups[0].shape[0]):
      Z = groups[0][c]
      cand1[c] = self.cache[Z]
    for c in range(groups[1].shape[0]):
      Z = groups[1][c]
      cand2[c] = self.cache[Z]
       
    return cand1, cand2 

  def train_groups(self, groups):
    ''' Prep and then run the batch jobs for the current tournaments. '''
    status = True if self.running_jobs() == 0 else err_exit("prior jobs are still running")
    self.cleanup()
    self.prep_jobs(groups) 
    # run jobs:
    job_cmd = f'for job in {tempdir}/*.job; do bsub < $job 2> /dev/null; done'
    status = True if os.system(job_cmd) == 0 else err_exit('cannot launch jobs')
    logmsg('launched. waiting 10 sec...')
    time.sleep(10)
    njobs = self.running_jobs()
    status = True if njobs > 0 else err_exit("jobs failed to launch")
    while njobs > 0:
      logmsg(f'waiting for jobs to complete. {njobs} jobs running.')
      time.sleep(3)
      njobs = self.running_jobs()
    logmsg('all jobs are complete. updating cache.')
    nfiles = self.update_cache(groups)
    status = True if nfiles > 0 else err_exit('no output files were found')
    candidates1, candidates2 = self.select_cache(groups)
    return (candidates1, candidates2)
    
  def train(self, W):
    ''' Train is called by the main GA loop to compute the test error for each
        candidate parent in W, based on models derived from the dev set. '''
    nerr=np.zeros(len(W))
    i = 0
    for Z in W:
      logmsg(f"training Z: {Z}")
      cache_val = self.cache[tuple(Z)]
      if cache_val < Hypers.init_cache: # cache hit
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
    child = np.random.rand(Hypers.nchrom)
    for n in range(Hypers.nchrom):
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
    newpop[mutei,0] = self.dmutate(newpop[mutei,0], delta, len(Hypers.Hparams['batch_size']))
    newpop[mutei,1] = self.dmutate(newpop[mutei,1], delta, len(Hypers.Hparams['optim']))
    newpop[mutei,2] = self.dmutate(newpop[mutei,2], delta, len(Hypers.Hparams['learn_rate']))
    return(newpop)

  def fresh_pop():
    ''' Return the initial population. '''
    logmsg("building new population")
    # is there a numpy way to do this instead of looping?
    pop_size = len(Hypers.Hparams['batch_size']) * len(Hypers.Hparams['optim']) * len(Hypers.Hparams['learn_rate'])
    newPop = np.zeros((pop_size,Hypers.nchrom), dtype=np.int32)
    i = 0                                                                                                 
    for ibsize in range(len(Hypers.Hparams['batch_size'])):
      for iopt in range(len(Hypers.Hparams['optim'])):
        for ilr in range(len(Hypers.Hparams['learn_rate'])):
          Z = np.array([ibsize, iopt, ilr], dtype=np.int32)
          newPop[i,:] = Z
          i += 1                                                                                         
    return newPop    
    
  def search(self):
    ''' The central algorithm, which mutates the population, searching for the best combo of genes. '''
    oldpop = Hypers.fresh_pop()
    newpop = oldpop.copy()
    popsize = len(oldpop)
    logmsg(f"search->population size: {popsize}, total epochs: {self.genmax*self.gennum}")
    
    # if the cache is full, then there is no point in running the GA:
    if (self.cache_size > 0) & (self.cache_size == self.num_cached):
      logmsg("cache is full.  exiting.")
      return (np.unravel_index(np.argmin(self.cache, axis=None), self.cache.shape))
    
    for i in range(self.genmax):       
      for j in range(self.gennum):  
        group_list = []
        candidates1=oldpop[np.random.choice(len(oldpop), size=self.ncontestants, replace=False)] 
        candidates2=oldpop[np.random.choice(len(oldpop), size=self.ncontestants, replace=False)] 
        group_list.append(candidates1)
        group_list.append(candidates2)
        candidates1, candidates2 = self.train_groups(group_list)

        parent1=candidates1[np.argmin(self.train(candidates1)),:]
        parent2=candidates2[np.argmin(self.train(candidates2)),:]
        newpop[j,:]=self.crossbreed(parent1,parent2)
        
        argmin_cache = np.unravel_index(np.argmin(self.cache, axis=None), self.cache.shape)
        np.save(self.cache_filename, self.cache)
        logmsg("gen: "+str(i)+", pop: "+str(j)+", min cache: "+str(self.cache[argmin_cache])+", argmin cache: "+str(argmin_cache))
      newpop=self.mutate(newpop)  
      oldpop=newpop.copy()
      
    H = np.unravel_index(np.argmin(self.cache, axis=None), self.cache.shape)
    return(H)

