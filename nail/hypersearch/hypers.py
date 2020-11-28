'''
This code uses a genetic algorithm (GA) to do a search for the best hyperparameters of any PyTorch model.

Author: Scott Miller, Visiting Scientist at the Nonlinear A.I. Lab (NAIL) of NCSU
'''


import datetime
from glob import glob
import math
import numpy as np
import os
import re
import time
import timeit
from   nail.hnn.utils import *

archivedir  = 'Hypers.archive'
jobname = 'hstj'
max_time = 15 # max runtime minutes for each job
tempdir  = 'Hypers.temp'

def err_exit(msg):
  logmsg(f'ERROR: {msg}')
  exit(-1)

class Hypers():
  ''' This module contains a genetic algorithm (GA) which searches for the best combination of hyperparameters of any
      PyTorch nn.module. The nn.module is created and trained externally via a separate script, provided by the user. 
      The training is run on the cluster using the bsub command.  The training jobs are monitored using the bjobs command.

      The search() function saves each training job's result in a cache, and will never retrain a candidate once it's
      in the cache.  It reloads the cache file upon startup, and saves it to disk after each training tournament.

      The GA converges when all members of the population are cached.'''

  def __init__(self, 
               cache_filename="hypers_cache.npy", genmax = 15, genpop = 25, ncontestants = 10, mutpct = 0.4, mutscale = 0.5):
    self.genmax = genmax
    self.genpop = genpop
    self.ncontestants = ncontestants
    self.mutpct = mutpct
    self.mutscale = mutscale
    self.cache_size = 0
    self.num_cached = 0
    self.cache = None
    self.cache_filename = cache_filename
    #self.cache_filename = f'/content/gdrive/My Drive/hypers/{cache_filename}'
    self.load_cache(cache_filename)
    ''' auto-adjust the generational loop parameters: '''
    if (genmax is None) and (genpop is None):
      self.genmax = int(math.sqrt(self.cache_size)) + 1
      self.genpop = self.genmax
    elif genmax is None:
      self.genmax = (self.cache_size // genpop) + 1 
    elif genpop is None:
      self.genpop = (self.cache_size // genmax) + 1 

  init_cache = 9999999
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
      'sgd'],
    'neurons' : ['2','4','8','16','32','64','128'],
    'layers' : ['2','4','8','16','32']
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
        self.num_cached = len(self.cache[self.cache < self.init_cache])
        logmsg(f"total cache size: {self.cache_size}")
        logmsg(f"num cached: {self.num_cached}")
        logmsg(f"cache shape: {self.cache.shape}")
  
  def show_z(self, Z):
    ''' Debug-print the given chromosome. '''
    logmsg(f"batch size: {Hypers.Hparams['batch_size'][Z[0]]}")
    logmsg(f"opt func: {Hypers.Hparams['optim'][Z[1]]}")
    logmsg(f"learn rate: {Hypers.Hparams['learn_rate'][Z[2]]}")
    
  def running_jobs(self):
    ''' Return the number of currently running jobs.'''
    try:
      stream = os.popen(f'bjobs | grep {jobname}  | wc -l 2> /dev/null')
                        #stdout=subprocess.PIPE, stderr=subpress.PIPE)
    except:
      err_exit('cannot run bjobs')
    rval = int(stream.read().rstrip('\n'))
    return rval
 
  def cleanup(self):
    ''' Archive then remove output files from prior jobs. '''
    global tempdir
    tstamp  = str(datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
    os.system(f'mkdir -p {tempdir}')
    status = True if os.path.isdir(tempdir) else err_exit(f'invalid tempdir "{tempdir}"')
    zname = f'{archivedir}/snapshot_logs_{tstamp}.zip'
    if len(glob(f'{tempdir}/*')) > 0:
      logmsg(f'archiving tempdir {tempdir} into zip file {zname}')
      status = True if os.system(f'zip {zname} {tempdir}/* > /dev/null') == 0 else err_exit('cannot zip snapshot files')
      status = True if os.system(f'rm -f {tempdir}/*') == 0 else err_exit('cannot rm temp files')
    return status

  def jobswait(self):
    ''' Wait until there are no more running jobs, and then return. '''
    njobs = self.running_jobs()
    while njobs > 0:
      logmsg(f'{njobs} running')
      time.sleep(3)
      njobs = self.running_jobs()
    return 0   

  def prep_job(Z, gid, cid):
    ''' Write a single .job file. '''
    zstr = ''
    for i in range(len(Z)):
      zstr += str(Z[i])
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
      ofile.writelines(f'#BSUB -J {jobname}\n')
      ofile.writelines(f'#BSUB -o {oname}\n')
      ofile.writelines(f'#BSUB -e {ename}\n')
      ofile.writelines(f'{cwd}/hypers.sh --learn_rate {lr} --optim {optf} --batch_size {bsize} --gid {gid} --cid {cid}\n')

    return 0

  def prep_all():
    ''' Prepare the .job parameter files for the entire search space.'''
    pop = Hypers.init_pop()
    for cid in range(len(pop)):  
      Z = pop[cid]
      Hypers.prep_job(Z, group_id=0, contestant_id=cid)
    logmsg(f'prepped {len(pop)} jobs for launch')

    return len(pop)

  def prep_jobs(self, groups):
    ''' Prepare the .job parameter files for each uncached candidate in groups. '''
    total = 0
    for group_id in range(len(groups)):
      W = groups[group_id]
      contestant_id = 0
      for Z in W:
        cache_val = self.cache[tuple(Z)]
        if cache_val == self.init_cache: # no cache hit
          logmsg(f'prepping. Z: {Z}')
          Hypers.prep_job(Z, group_id, contestant_id)
          total += 1
        else:
          logmsg(f'cache hit. Z: {Z}')
        contestant_id += 1
    logmsg(f'prepped {total} jobs for launch')

    return total
    
  def update_cache(self, groups):
    ''' Parse the output files from the recently completed jobs, and collect the loss values from each job.
        Update the cache with the results. Return the number of files parsed.
        oname = f'{tempdir}/hypers_group{group_id}_Z{Z}.out' '''
    cpatt = re.compile('^CID: ([0-9\.]+).*$')
    gpatt = re.compile('^GID: ([0-9\.]+).*$')
    hpatt = re.compile('^HOSTNAME: (.+)$')
    lpatt = re.compile('^LOSS: ([0-9\.]+).*$')
    nfiles = 0
    for fname in glob(f'{tempdir}/hypers_group*_Z*.out'):
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
          logmsg(f'gid: {gid} cid: {cid} loss: {loss} hostname: {hostname}')
          err_exit(f'file {fname} does not parse')
        Z = groups[gid][cid]
        logmsg(f'gid: {gid} cid: {cid}, updating cache[{Z}] = {loss} on hostname {hostname}')
        self.cache[tuple(Z)] = loss

    return nfiles

  def select_cache(self, groups):
    ''' Return the cache values for the groups, split into two groups, one for 
        each tournament. '''
    cand1, cand2 = (np.zeros((groups[0].shape[0],)), np.zeros((groups[0].shape[0],)))
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
    job_cmd = f'for job in {tempdir}/*.job; do bsub < $job > /dev/null; done'
    status = True if os.system(job_cmd) == 0 else err_exit('cannot launch jobs')
    logmsg('launched. waiting 10 sec...')
    time.sleep(10)
    njobs = self.running_jobs()
    status = True if njobs > 0 else err_exit("jobs failed to launch")
    while njobs > 0:
      logmsg(f'# jobs running: {njobs}')
      time.sleep(3)
      njobs = self.running_jobs()
    logmsg('all jobs are complete. updating cache.')
    nfiles = self.update_cache(groups)
    status = True if nfiles > 0 else err_exit('no output files were found')
    
    return nfiles

  def check_jobs(self, groups):
    ''' Ensure all completed jobs were parsed successfully by checking that all 
        group members are now in the cache:'''
    uncached = groups[0][groups[0] == self.init_cache]
    logmsg(f'uncached 1: {uncached}')
    uncached = np.concatenate((uncached, groups[1][groups[1] == self.init_cache]))
    logmsg(f'uncached 2: {uncached}')
    if len(uncached) > 0:
      logmsg('no output file was found for the following candidates:')
      for i in range(len(uncached)):
        logmsg(f'  {uncached[i]}')
      err_exit('see above')

    return True
    
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
        err_exit(f'{nprepped} jobs prepped but only {ncompleted} completed')
    self.check_jobs(groups)
    tournament1, tournament2 = self.select_cache(groups)

    return (tournament1, tournament2)

  def offspring(self, parent1, parent2):      
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

  def init_pop():
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
    
  def best(self):
    ''' Return the chromosome with the lowest cost, from the cache.'''
    H = np.unravel_index(np.argmin(self.cache, axis=None), self.cache.shape)
    return H

  def save_cache(self):
    ''' First archive the existing cache file, and then save the cache.'''
    tstamp  = str(datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
    os.system(f'mkdir -p {archivedir}')
    zname = f'{archivedir}/snapshot_cache_{tstamp}.zip'
    status = True if os.system(f'zip {zname} {self.cache_filename} > /dev/null') == 0 else logmsg('cannot archive cache file')
    if np.save(self.cache_filename, self.cache): 
      err_exit('cannot save cache')

  def cached_of(self, population):
    ''' Return the number of candidates in population that are already cached.'''
    num = 0
    for i in range(len(population)):
      Z = population[i]
      num += 1 if self.cache[tuple(Z)] < self.init_cache else 0
    return num

  def find(pop, Z):
    ''' Return the index of Z within pop.'''
    for i in range(len(pop)):
      if np.array_equal(pop[i], Z):
        return i
    return None
 
  def search(self):
    ''' The genetic algorithm (GA), which competes, offsprings and mutates the population, 
        searching for the best combo of chromosomes. '''
    oldpop = Hypers.init_pop()
    newpop = oldpop.copy()
    popsize = len(oldpop)
    logmsg(f"search->population size: {popsize}, total epochs: {self.genmax*self.genpop}")
    
    # if the cache is full, then there is no point in running the search:
    if (self.cache_size > 0) & (self.cache_size == self.num_cached):
      logmsg("cache is full.  exiting.")
      return self.best()
    
    Z = np.array([6,0,1], dtype=int)
    for igen in range(self.genmax):       
      for ipop in range(self.genpop):  
        group_list = []
        candidates=oldpop[np.random.choice(len(oldpop), size=(2*self.ncontestants), replace=False)] 
        tournament1=candidates[:self.ncontestants]
        tournament2=candidates[self.ncontestants:]
        group_list.append(tournament1)
        group_list.append(tournament2)
        losses1, losses2 = self.train_groups(group_list)
        parent1 = tournament1[np.argmin(losses1), :]
        parent2 = tournament2[np.argmin(losses2), :]
 
        ''' replace the poorest performer with the offspring of the two winners: '''
        losses = np.concatenate((losses1,losses2))
        poorest = candidates[np.argmax(losses), :]
        inew = Hypers.find(newpop, poorest)
        '''
        iloc = np.flatnonzero((newpop == poorest).all(1))
        try:
          inew = iloc[0]
        except:
          logmsg(f'iloc: {iloc}')
          logmsg(f'poorest: {poorest}')
          logmsg(f'len(newpop): {len(newpop)} newpop.shape: {newpop.shape}')
          err_exit('exception dereferencing iloc[0]')
        finally:
          newpop[inew,:] = self.offspring(parent1, parent2)
        '''
        logmsg(f'poorest: {poorest} inew: {inew} newpop[inew]: {newpop[inew]}')
        newpop[inew,:] = self.offspring(parent1, parent2) if inew is not None else err_exit(f'poorest {poorest} not found')
        
        logmsg(f'cache shape: {self.cache.shape}, num cached: {len(np.where(self.cache < self.init_cache))}')
        argmin_cache = np.unravel_index(np.argmin(self.cache, axis=None), self.cache.shape)
        logmsg(f"gen: {igen}, pop: {ipop}, min cache: {self.cache[argmin_cache]}, argmin cache: {argmin_cache}")

      newpop=self.mutate(newpop)  
      oldpop=newpop.copy()
      if self.cached_of(newpop) == popsize:
        logmsg('convergence.  all newpop is cached.')
        exit(0)
      
    return self.best()

  def collect_all(self):
    ''' Build a single group containing all candidates, and then use it to update the 
        cache from a collection of output files. '''
    pop = Hypers.init_pop()
    groups = [pop]
    nfiles = self.update_cache(groups)
    self.save_cache()
    
    return nfiles
