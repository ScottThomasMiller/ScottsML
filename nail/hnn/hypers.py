'''
This code uses a genetic algorithm (GA) to do a search for the best hyperparameters of any PyTorch model.

Author: Scott Miller, Visiting Scientist at the Nonlinear A.I. Lab (NAIL) of NCSU
'''

import configparser
import datetime
from glob import glob
import math
import numpy as np
import os
import re
import time
import timeit
import torch
from   utils import *

archivedir  = 'Hypers.archive'
tempdir  = 'Hypers.temp'
jobname = 'hypersjob'
max_time = '120' # max runtime minutes for each job
queue_name = 'short'
num_cores = 4 # cores per training job
converge_pct = 0.05 


def err_exit(msg):
  logmsg(f'ERROR: {msg}')
  exit(-1)

 
class Hypers():
  ''' This module contains a genetic algorithm (GA) which searches for the best combination of hyperparameters of any
      PyTorch nn.module. The nn.module is created and trained externally via a separate script, provided by the user. 
      The training is run on the cluster using the bsub command.  The training jobs are monitored using the bjobs command.

      The search() function saves each training job's result in a cache, and will never retrain a candidate once it's
      in the cache.  It reloads the cache file upon startup, and saves it to disk after each training tournament.

      The GA converges when two successive generations are not different enough.
  '''
  def load_config(self, config_filename):
    ''' Read and return the config settings.'''
    logmsg(f'reading config file {config_filename}')
    self.config = configparser.ConfigParser()
    self.config.read(config_filename)
    for k in self.config['DEFAULT']:
      val = self.config['DEFAULT'][k]
      logmsg(f'  {k}: {val}')

  def __init__(self, config_filename = 'hypers.ini'):
    self.load_config(config_filename)
    self.max_gen = int(self.config['DEFAULT']['max_gen'])
    self.repl_per_gen = int(self.config['DEFAULT']['repl_per_gen'])
    self.trial_size = int(self.config['DEFAULT']['trial_size'])
    self.mutn_pct = float(self.config['DEFAULT']['mutn_pct'])
    self.mutn_scale = float(self.config['DEFAULT']['mutn_scale'])
    self.save_dir = self.config['DEFAULT']['save_dir']
    cfname = self.config['DEFAULT']['cache_filename']
    self.cache_filename = f"{self.save_dir}/{cfname}"
    self.init_pop_size = int(self.config['DEFAULT']['init_pop_size'])

    self.cache = None
    self.visited = None
    self.archivepath = f'{self.save_dir}/{archivedir}'
    self.temppath = f'{self.save_dir}/{tempdir}'
    #self.cache_filename = f'/content/gdrive/My Drive/hypers/{cache_filename}'
    self.load_cache(self.cache_filename)
    ''' auto-adjust the generational loop parameters: '''
    if (self.max_gen is None) and (self.repl_per_gen is None):
      self.max_gen = int(math.sqrt(self.cache.size)) + 1
      self.repl_per_gen = self.max_gen
    elif self.max_gen is None:
      self.max_gen = (self.cache.size // self.repl_per_gen) + 1 
    elif self.repl_per_gen is None:
      self.repl_per_gen = (self.cache.size // self.max_gen) + 1 
    logmsg(f'gen parameters post-adjust->max_gen: {self.max_gen} repl_per_gen: {self.repl_per_gen} total: {self.max_gen*self.repl_per_gen}')

  ngenes = 5  #number of genes/hyperparameters
  init_cache = 999999999.0

  ''' The following is the lookup array for all chromosome values.  The GA populations
      will be arrays of ngenes-length vectors.  Each position in each vector is an
      index into one of the following lists: '''
  Hparams = {
    'batch_size' : ['1','2','4','8','16'],
    'optim' : [
      'adadelta',
      'adagrad',
      'adam',
      'adamax',
      'adamw',
      'asgd',
      'rmsprop',
      'rprop',
      'sgd'],
    'learn_rate' : ['1e-5',' 1e-4',' 1e-3',' 1e-2',' 1e-1'],
    'neurons' : ['1','2','4','8','16'],
    'layers' : ['1','2','4','8','16']
    }

  ''' future hyperparameters to consider: 
  'cost_func' : [  
    'mean_squared_error',
    'mean_squared_logarithmic_error',
    'categorical_hinge',
    'categorical_crossentropy'],
  'act_func' : [
      'relu',
      'sigmoid',
      'tanh'
  ]
  '''

  def load_pop(self, pop_filename):
    try:
      logmsg(f"loading population file {pop_filename}")
      return np.load(pop_filename)
    except:
      logmsg("population load failed.")
      return self.sample_pop()

  def load_cache(self, cache_filename):
    try:
      logmsg(f"loading cache file {self.cache_filename}")
      self.cache = np.load(self.cache_filename)      
    except:
      logmsg("load failed. initializing new cache.")
      nbs = len(Hypers.Hparams['batch_size'])
      nopt = len(Hypers.Hparams['optim'])
      nlr = len(Hypers.Hparams['learn_rate'])
      nnr = len(Hypers.Hparams['neurons'])
      nlyr = len(Hypers.Hparams['layers'])
      self.cache = np.full((nbs,nopt,nlr,nnr,nlyr), Hypers.init_cache, dtype=float)
    finally:
        num_cached = self.cache[self.cache < Hypers.init_cache].size
        self.visited = np.zeros(self.cache.shape, dtype=int)
        logmsg(f"total cache size: {self.cache.size}")
        logmsg(f"num cached: {num_cached}")
        logmsg(f"cache shape: {self.cache.shape}")
  
  def show_z(self, Z):
    ''' Debug-print the given chromosome. '''
    logmsg(f"batch size: {Hypers.Hparams['batch_size'][Z[0]]}")
    logmsg(f"opt func: {Hypers.Hparams['optim'][Z[1]]}")
    logmsg(f"learn rate: {Hypers.Hparams['learn_rate'][Z[2]]}")
    logmsg(f"neurons: {Hypers.Hparams['neurons'][Z[3]]}")
    logmsg(f"layers: {Hypers.Hparams['layers'][Z[4]]}")
    
  def running_jobs(self):
    ''' Return the number of currently running jobs.'''
    try:
      stream = os.popen(f'bjobs | grep {jobname} | grep RUN | wc -l 2> /dev/null')
                        #stdout=subprocess.PIPE, stderr=subpress.PIPE)
    except:
      err_exit('cannot run bjobs')
    rval = int(stream.read().rstrip('\n'))
    return rval
 
  def cleanup(self):
    ''' Archive then remove output files from prior jobs. '''
    tstamp  = str(datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
    os.system(f'mkdir -p {self.temppath}')
    status = True if os.path.isdir(self.temppath) else err_exit(f'invalid tempdir "{self.temppath}"')

    zname = f'{self.archivepath}/snapshot_logs_{tstamp}.zip'
    if len(glob(f'{self.temppath}/*')) > 0:
      logmsg(f'archiving tempdir {self.temppath} into zip file {zname}')
      status = True if os.system(f'zip {zname} {self.temppath}/* > /dev/null') == 0 else err_exit('cannot zip snapshot files')
      status = True if os.system(f'rm -f {self.temppath}/*') == 0 else err_exit('cannot rm temp files')
    return status

  def jobswait(self):
    ''' Wait until there are no more running jobs, and then return. '''
    njobs = self.running_jobs()
    while njobs > 0:
      logmsg(f'{njobs} running')
      time.sleep(3)
      njobs = self.running_jobs()
    return 0   

  def prep_job(self, Z, gid, cid, seq):
    ''' Write a single .job file. '''
    zstr = ''
    for i in range(len(Z)):
      zstr += str(Z[i])
    jname = f'{self.temppath}/hypers_group{gid}_Z{zstr}.job'
    oname = f'{self.temppath}/hypers_group{gid}_Z{zstr}.out'
    ename = f'{self.temppath}/hypers_group{gid}_Z{zstr}.err'
    cwd = os.getcwd()
    bsize = Hypers.Hparams['batch_size'][Z[0]]
    optf = Hypers.Hparams['optim'][Z[1]]
    lr = Hypers.Hparams['learn_rate'][Z[2]]
    neurons = Hypers.Hparams['neurons'][Z[3]]
    layers = int(Hypers.Hparams['layers'][Z[4]])
    status = 0
    dim = ""
    for n in range(layers):
      dim += " " + neurons
    
    with open(jname, 'w') as ofile:
      ofile.writelines(f'#!/bin/tcsh\n')
      ofile.writelines(f'#BSUB -J {jobname}{seq}\n')
      ofile.writelines(f'#BSUB -n {num_cores}\n')
      ofile.writelines(f'#BSUB -W {max_time}\n')
      #ofile.writelines(f'#BSUB -R span[hosts=1]\n')
      ofile.writelines(f'#BSUB -q {queue_name}\n')
      ofile.writelines(f'#BSUB -o {oname}\n')
      ofile.writelines(f'#BSUB -e {ename}\n')
      ofile.writelines(f'{cwd}/hypers.sh --learn_rate {lr} --optim {optf} --batch_size {bsize} --gid {gid} --cid {cid} --hidden_dim {dim}\n')
      logmsg(f'job filename: {jname}')

    return 0

    return len(pop)

  def prep_all(self, skip_cached=False):
    ''' Prepare the .job parameter files for the entire initial population.  If
        skip_cached is True then prep only the uncached entries.'''
    self.cleanup()
    nprep = 0
    pop = self.init_pop()
    for cid in range(len(pop)):  
      Z = pop[cid]
      cache_val = self.cache[tuple(Z)]
      if (cache_val < Hypers.init_cache) and skip_cached: 
        continue
      self.prep_job(Z, gid=0, cid=nprep, seq=cid)
      nprep += 1
    logmsg(f'prepped {nprep} jobs for launch')

    return len(pop)

  def prep_jobs(self, groups):
    ''' Prepare the .job parameter files for each uncached candidate in groups. '''
    total = 0
    for group_id in range(len(groups)):
      W = groups[group_id]
      contestant_id = 0
      for Z in W:
        self.visited[tuple(Z)] += 1
        cache_val = self.cache[tuple(Z)]
        if cache_val >= Hypers.init_cache: # no cache hit
          self.prep_job(Z, group_id, contestant_id, seq=total)
          total += 1
        contestant_id += 1
       
    logmsg(f'prepped {total} jobs for launch')

    return total
    
  def update_cache(self, groups):
    ''' Parse the output files from the recently completed jobs, and collect the loss values from each job.
        Update the cache with the results. Return the number of files parsed.
        oname = f'{self.temppath}/hypers_group{group_id}_Z{Z}.out' '''
    cpatt = re.compile('^CID: ([0-9\.]+).*$')
    gpatt = re.compile('^GID: ([0-9\.]+).*$')
    hpatt = re.compile('^HOSTNAME: (.+)$')
    lpatt = re.compile('^LOSS: ([0-9\.]+).*$')
    nfiles = 0
    ngroups = len(groups)
    for fname in glob(f'{self.temppath}/hypers_group*_Z*.out'):
      nfiles += 1
      Z, gid, cid, loss, hostname = None, None, None, None, None
      logmsg(f'inspecting file {fname}')
      with open(fname,'r') as infile:
        for line in infile.readlines():
          cmatch = cpatt.match(line)
          gmatch = gpatt.match(line)
          hmatch = hpatt.match(line)
          lmatch = lpatt.match(line)
          if cmatch is not None:
            cid = int(cmatch.group(1)) 
          elif gmatch is not None:
            gid = int(gmatch.group(1)) 
          elif hmatch is not None:
            hostname = hmatch.group(1)
          elif lmatch is not None:
            loss = float(lmatch.group(1)) 

        if (gid is None) or (cid is None) or (loss is None) or (hostname is None):
          logmsg(f'WARNING: file {fname} does not parse')
          logmsg(f'gid: {gid} cid: {cid} loss: {loss} hostname: {hostname}')
        else:
          #Z = groups[gid][cid] if gid <= ngroups else groups[0][cid]
          Z = groups[gid][cid] 
          logmsg(f'gid: {gid} cid: {cid}, updating cache[{Z}] = {loss} from hostname {hostname}')
          self.cache[tuple(Z)] = loss

    return nfiles

  def select_cache(self, groups):
    ''' Return the cache values for the groups, split into two groups, one for 
        each tournament. '''
    #cand1, cand2 = (np.zeros((groups[0].shape[0],)), np.zeros((groups[0].shape[0],)))
    cand1, cand2 = (np.zeros((groups[0].shape[0],)), np.zeros((groups[1].shape[0],)))
    for c in range(groups[0].shape[0]):
      Z = groups[0][c]
      cand1[c] = self.cache[tuple(Z)]
    for c in range(groups[1].shape[0]):
      Z = groups[1][c]
      cand2[c] = self.cache[tuple(Z)]
       
    return cand1, cand2 

  def run_jobs(self, groups):
    ''' Launch the prepped jobs for the given groups, and then wait for them
        all to complete. Process the output files to update the cache.  
        Return the number of output files processed'''
    job_cmd = f'for job in {self.temppath}/*.job; do bsub < $job > /dev/null; done'
    status = True if os.system(job_cmd) == 0 else err_exit('cannot launch jobs')
    logmsg('launched. waiting 30 sec...')
    time.sleep(30)
    njobs = self.running_jobs()
    status = True if njobs > 0 else logmsg("no jobs are running")
    while njobs > 0:
      logmsg(f'# jobs running: {njobs}')
      time.sleep(3)
      njobs = self.running_jobs()
    nfiles = self.update_cache(groups)
    if nfiles == 0:
      logmsg('WARNING: no output files were found')
    
    return nfiles

  def check_jobs(self, groups):
    ''' Ensure all completed jobs were parsed successfully by checking that all 
        group members are now in the cache:'''
    uncached = groups[0][groups[0] is None]
    uncached = np.concatenate((uncached, groups[1][groups[1] is None]))
    if len(uncached) > 0:
      logmsg('WARNING: no output file was found for the following candidates:')
      for i in range(len(uncached)):
        logmsg(f'  {uncached[i]}')

    return True
    
  def vectorZ(Zimg, popshape):
    ''' Return the int-vector Z of the integer image, created by copying its digits. 
        E.g.: 123 -> [1, 2, 3] '''
    sZimg = Zimg
    zlen = popshape[1] 
    Z = np.ones((zlen), dtype=int)
    for i in range(zlen):
      power = (zlen - i) - 1
      tens = 10 ** power
      Z[i] = Zimg // tens
      Zimg -= Z[i] * tens
      power -= 1

    return Z

  def imageZ(Z):
    ''' Return the integer image of the int-vector Z, created by concatenating its digits. 
        E.g.: [1, 2, 3] -> 123 '''
    zs = Z.astype(str)
    Zstr = ''
    for i in range(len(zs)):
      Zstr += zs[i]
   
    return int(Zstr)
    
  def __stubtest__(self, groups):
    ''' This is a stub for testing the GA.  It uses each candidate's numeric image to
        quickly compute a "fitness score". '''
    cands = []
    for g in range(len(groups)): 
      cands.append(np.zeros((groups[g].shape[0],), dtype=int))
      for c in range(groups[g].shape[0]):
        cands[g][c] = Hypers.imageZ(groups[g][c])
        Z = groups[g][c]
        self.cache[tuple(Z)] = cands[g][c]

    return (cands[0], cands[1])

  def train_groups(self, groups):
    ''' Prep and then run the batch jobs for the given groups. Return the training results 
        vectors for both groups, containing the loss value for each candidate.'''
    status = True if self.running_jobs() == 0 else err_exit("prior jobs are still running")
    self.cleanup()
    nprepped = self.prep_jobs(groups)
    if nprepped > 0:
      ncompleted = self.run_jobs(groups)
      self.save_cache()
      if ncompleted != nprepped:
        logmsg(f'WARNING: {nprepped} jobs prepped but only {ncompleted} completed')
    self.check_jobs(groups)
    tournament1, tournament2 = self.select_cache(groups)

    return (tournament1, tournament2)

  def offspring(self, parent1, parent2):      
    ''' Return the child of both parents, taking each gene from only one parent at random. '''
    #child = np.random.rand(Hypers.ngenes)
    child = np.zeros(parent1.shape, dtype=int)
    for n in range(Hypers.ngenes):
      rando = np.random.rand(1)
      #child[n] = rando*parent1[n] + (1-rando)*parent2[n]
      child[n] = parent1[n] if rando < 0.5 else parent2[n]
      
    return(child)

  def dmutate(self, Muteys, delta, cap_value):    
    ''' Translate a continuous mutation delta into an integer hyperparameter index. '''
    New = Muteys.astype(np.float64) + delta
    Muteys[New >= 0] = np.rint(New[New >= 0]) % cap_value
    Muteys[New < 0] = cap_value - 1;
    return (Muteys)
    
  def mutate(self, newpop):
    ''' Return the new, mutated population. '''
    nmutations = math.ceil(len(newpop)*self.mutn_pct)
    mutei=np.random.choice(len(newpop), nmutations, replace=False, p=None)
    delta = np.random.normal(0,self.mutn_scale)
    newpop[mutei,0] = self.dmutate(newpop[mutei,0], delta, len(Hypers.Hparams['batch_size']))
    delta = np.random.normal(0,self.mutn_scale)
    newpop[mutei,1] = self.dmutate(newpop[mutei,1], delta, len(Hypers.Hparams['optim']))
    delta = np.random.normal(0,self.mutn_scale)
    newpop[mutei,2] = self.dmutate(newpop[mutei,2], delta, len(Hypers.Hparams['learn_rate']))
    delta = np.random.normal(0,self.mutn_scale)
    newpop[mutei,3] = self.dmutate(newpop[mutei,3], delta, len(Hypers.Hparams['neurons']))
    delta = np.random.normal(0,self.mutn_scale)
    newpop[mutei,4] = self.dmutate(newpop[mutei,4], delta, len(Hypers.Hparams['layers']))
    return(newpop)

  def init_pop(self):
    ''' Return the total possible population. '''
    logmsg("init_pop()")
    # is there a numpy way to do this instead of looping?
    # use np.ogrid or grid instead?
    nbs = len(Hypers.Hparams['batch_size'])
    nopt = len(Hypers.Hparams['optim'])
    nlr = len(Hypers.Hparams['learn_rate'])
    nnr = len(Hypers.Hparams['neurons'])
    nlyr = len(Hypers.Hparams['layers'])
    pop_size = nbs * nopt * nlr * nnr * nlyr
    pop0 = np.zeros((pop_size, Hypers.ngenes), dtype=np.int32)
    i = 0                                                                                                 
    for ibsize in range(nbs):
      for iopt in range(nopt):
        for ilr in range(nlr):
          for inr in range(nnr):
            for ilyr in range(nlyr):
              Z = np.array([ibsize, iopt, ilr, inr, ilyr], dtype=np.int32)
              pop0[i,:] = Z
              i += 1                                                                                         

    return pop0
    
  def sample_pop(self):
    ''' Return a random sample of the total possible population. '''
    pop0 = self.init_pop()
    pop0 = pop0[np.random.choice(len(pop0), size=(self.init_pop_size), replace=False)]

    return pop0    
    
  def best(self):
    ''' Return the chromosome with the lowest cost, from the cache.'''
    H = np.unravel_index(np.argmin(self.cache, axis=None), self.cache.shape)

    return H

  def save_pop(self, pop, pop_filename):
    ''' Archive the population.'''
    os.system(f'mkdir -p {self.archivepath}')
    filepath = f'{self.archivepath}/{pop_filename}'
    if np.save(filepath, pop):
      err_exit(f'cannot save population to file {filepath}')

  def save_cache(self):
    ''' First archive the existing cache file, and then save the cache.'''
    tstamp  = str(datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
    os.system(f'mkdir -p {self.archivepath}')
    zname = f'{self.archivepath}/snapshot_cache_{tstamp}.zip'
    #status = True if os.system(f'zip {zname} {self.cache_filename} > /dev/null') == 0 else logmsg(f'WARNING: cannot archive cache file {self.cache_filename} into zip file {zname}.')
    status = True if os.system(f'zip {zname} {self.cache_filename}') == 0 else logmsg(f'WARNING: cannot archive cache file {self.cache_filename} into zip file {zname}.')
    if np.save(self.cache_filename, self.cache): 
      err_exit(f'cannot save cache to file {self.cache_filename}')

  def cached_of(self, population):
    ''' Return the number of candidates in population that are already cached.'''
    num = 0
    for i in range(len(population)):
      Z = population[i]
      num += 1 if self.cache[tuple(Z)] is not None else 0

    return num

  def find(pop, Z):
    ''' Return the index of Z within pop.'''
    for i in range(len(pop)):
      if np.array_equal(pop[i], Z):
        return i

    return None
 
  def find_weakests(candidates, losses, pop):
    ''' Return the iloc of the candidate with the weakests score, and which exists in
        pop.  If no such animal, then return None. '''
    cands = candidates.copy()
    ls = losses.copy()
    while len(cands) > 0:
      imax = np.argmax(ls)
      weakests = cands[imax, :]
      iloc = np.flatnonzero((pop == weakests).all(1))
      if len(iloc) > 0:
        return iloc
      else:
        # remove candidate and try again.
        cands = np.delete(cands, imax, axis=0) 
        ls = np.delete(ls, imax, axis=0) 
    return None

  def min_pop(pop):
    ''' Return the min value from the population, and the number of unique entries. '''
    sorted, nuq = Hypers.sort_pop(pop)
    return sorted[0], nuq

  def max_pop(pop):
    ''' Return the max value from the population, and the number of unique entries. '''
    sorted, nuq = Hypers.sort_pop(pop)
    return sorted[-1], nuq

  def sort_pop(pop):
    ''' Return pop sorted by chromosome in numeric-image order, and the number of 
        unique entries. '''
    ipop = np.ones((len(pop),), dtype=int)
    sorted = np.ones(pop.shape, dtype=int)
    for i in range(len(pop)):
      intZ = Hypers.imageZ(pop[i])
      ipop[i] = intZ
    spop = np.sort(ipop, axis=0)
    for i in range(len(spop)):
      sorted[i] = Hypers.vectorZ(spop[i], pop.shape)

    return sorted, len(np.unique(spop))

  def matching(equals):
    ''' Return the number of boolean vectors that are completely True. '''
    nsame = 0
    for i in range(len(equals)):
      nsame += 1 if (sum(equals[i]) == len(equals[i])) else 0

    return nsame

  def converged(oldpop, newpop):
    ''' Return True if old and new populations are identical or nearly so. Deprecated in favor
        of num unique. '''
    oldsorted, nuqold = Hypers.sort_pop(oldpop)
    newsorted, nuqnew = Hypers.sort_pop(newpop)
    if np.array_equal(oldsorted, newsorted):
      logmsg(f'converged()->identical populations')
      return True
    equals = np.equal(oldsorted, newsorted)
    zeros = np.zeros(oldsorted.shape)
    ones = np.ones(oldsorted.shape)
    #nsame = Hypers.matching(equals)
    nsame = 0
    for i in range(len(oldpop)):
      nsame += 1 if np.array_equal(oldpop[i], newpop[i]) else 0
    logmsg(f'pop size: {len(oldsorted)} nsame: {nsame} ')
    #if nsame >= converge_pct*len(oldsorted):
    if nsame >= len(oldsorted):
      return True

    return False
  
  def select_indexes(self, pop):
    ''' Return a list of the indices of both the cached and uncached chromosomes in the given population. '''
    cachedA = []
    uncachedA = []
    for i in range(len(pop)):
      Z = pop[i]
      if self.cache[tuple(Z)] < Hypers.init_cache: # cache hit
        cachedA.append(i)
      else:
        uncachedA.append(i)

    return np.asarray(cachedA, dtype=int), np.asarray(uncachedA, dtype=int)

  def select_candidates(self, oldpop):
    ''' Choose the next batch of chromosomes for the competition.  Give priority to uncached entries. '''
    icached, iuncached = self.select_indexes(oldpop)
    logmsg(f'len(icached): {len(icached)} len(iuncached): {len(iuncached)}')
    if len(iuncached) == 0:
      return oldpop[np.random.choice(len(oldpop), size=(2*self.trial_size), replace=False)] 
    else:
      # use up the uncached first:
      nadd = len(iuncached) if len(iuncached) < 2*self.trial_size else 2*self.trial_size
      candidates=oldpop[iuncached][np.random.choice(len(oldpop[iuncached]), size=(nadd), replace=False)] 
      # finish filling up the list with cached entries, if necessary:
      nadd = 2*self.trial_size - len(iuncached)
      if nadd > 0:
        add = oldpop[icached][np.random.choice(len(oldpop[icached]), size=(nadd), replace=False)] 
        candidates = np.concatenate((candidates, add))

      return candidates

  def search(self):
    ''' The genetic algorithm (GA), which competes, offsprings and mutates the population, 
        searching for the best combo of chromosomes. '''
    oldpop = self.load_pop('hypers_pop.npy')
    newpop = oldpop.copy()
    popsize = len(oldpop)
    tstamp  = str(datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
    popseq = 0
    self.save_pop(oldpop, f'hyperspop_{tstamp}_{popseq:04d}.npy')
    logmsg(f"search->population size: {popsize}, total epochs: {self.max_gen*self.repl_per_gen}")

    for igen in range(self.max_gen):       
      popseq += 1
      for ipop in range(self.repl_per_gen):  
        group_list = []
        candidates = self.select_candidates(oldpop)

        ''' split the list of candidates into two groups: '''
        tournament1=candidates[:self.trial_size]
        tournament2=candidates[self.trial_size:]
        group_list.append(tournament1)
        group_list.append(tournament2)

        ''' evaluate all candidates and collect results: '''
        losses1, losses2 = self.train_groups(group_list)
        #losses1, losses2 = self.__stubtest__(group_list)
        parent1 = tournament1[np.argmin(losses1), :]
        parent2 = tournament2[np.argmin(losses2), :] if len(losses2) > 0 else parent1
 
        ''' replace the weakest performer(s) with the offspring of the two winners: '''
        losses = np.concatenate((losses1,losses2))
        weakests = Hypers.find_weakests(candidates, losses, newpop)
        if weakests is None:
          logmsg(f'WARNING: weakest not found')
        else:
          baby = self.offspring(parent1, parent2)
          newpop[weakests, :] = baby       
          #baby2D = np.asarray([baby])
          #np.delete(oldpop, weakests, axis=0)
          #np.append(oldpop, baby2D, axis=0)
        
        argmin_cache = np.unravel_index(np.argmin(self.cache, axis=None), self.cache.shape)
        minpopZ, nuq = Hypers.min_pop(newpop)
        epoch = (igen*self.repl_per_gen) + ipop
        mincacheval = self.cache[argmin_cache]
        minpopval = self.cache[tuple(minpopZ)] 
        logmsg(f"gen: {igen} pop: {ipop} epoch: {epoch}, cache:: Z: {argmin_cache} val: {mincacheval:1.2E} " + 
               f"pop:: Z: {minpopZ} val: {minpopval:1.2E} nuq(new): {nuq} visited: {self.visited[self.visited > 0].size}")

      #if nuq == 1:
      if Hypers.converged(oldpop, newpop):
        self.save_pop(oldpop, f'hyperspop_{tstamp}_{popseq:04d}.npy')
        logmsg('convergence. clean exit.')
        logmsg(f'\nbest of new pop: {minpopZ}')
        self.show_z(minpopZ)
        return self.best()
      self.save_pop(oldpop, f'hyperspop_{tstamp}_{popseq:04d}.npy')
      newpop = self.mutate(newpop)  
      oldpop = newpop.copy()
      
    logmsg('exhaustion. no convergence.')
    return self.best()

  def collect_all(self):
    ''' Build a single group containing all candidates, and then use it to update the 
        cache from a collection of output files. '''
    pop = self.init_pop()
    groups = [pop]
    nfiles = self.update_cache(groups)
    self.save_cache()
    
    return nfiles
