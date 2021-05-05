   
import glob
import os
import random 
import shutil
import utils
   
global folders, categories
folders = ['train','test','validate']
categories = ['EyeBlinks','Noise','JawClenches','ToeTaps']
   
class Instance():
  def __init__(self, fname, cat): 
   self.fname = fname
   self.cat = cat
   
# Define a class to categorize each instance
class Experiment(): 
  ''' Create an experiment by iterating over the category subfolders of dir,
      reading their filenames into memory, and then shuffling them.
  '''
  def __init__(self, dir): 
    print(f'dir: {dir}')
    self.dir = dir 
    self.all_insts = [] 
    for fname in glob.glob(f'{dir}/*'):
      self.all_insts.append(Instance(fname, cat))
    random.shuffle(self.all_insts) 
   
  # Method to remove a card from the deck 
  def popCard(self): 
   if len(self.all_insts) == 0: 
    return None
   else: 
    return self.all_insts.pop() 
   
def cleanup():
  utils.logmsg('cleanup')
  for fname in glob.glob('/Users/scottmiller/data/ACE/CreateML/experiment/**/*.*', recursive=True):
    utils.logmsg('  deleting file {0}'.format(fname))
    os.remove(fname)

if __name__ == "__main__":
  args = utils.get_args()
  random.seed(args.seed)
  cleanup()
  for cat in categories:
    traindir = f"/Users/scottmiller/data/ACE/CreateML/experiment/train/{cat}"
    testdir = f"/Users/scottmiller/data/ACE/CreateML/experiment/test/{cat}"
    validdir = f"/Users/scottmiller/data/ACE/CreateML/experiment/validate/{cat}"
    exp = Experiment(f'{args.save_dir}/{cat}')
    numtest, numvalid = 0, 0
   
    ''' copy the first 10 into test and valid folders, 5 each: '''
    for inst in exp.all_insts:
      sname = os.path.basename(inst.fname)
      if numtest < 5:
        dir = testdir
        numtest += 1
      elif numvalid < 5:
        dir = validdir
        numvalid += 1
      else:
        dir = traindir
      if not os.path.exists(dir):
        os.makedirs(dir)
      shutil.copy2(inst.fname, f'{dir}/')
      utils.logmsg(f'copied {sname} to {dir}')

   
