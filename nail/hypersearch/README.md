This is the hypers folder.  Hypers is a Python class which uses a genetic algorithm (GA) to perform a hyperparameter search for any PyTorch model, on the HPC cluster at NCSU.  The tournaments within each epoch are run in parallel on the cluster, yielding a huge speedup over traditional GA.  The GA converges when its population is completely cached.

The main file is hypers.py.  It contains the GA, and it uses bsub and bjobs to launch and monitor the training jobs.  Because it might take hours to complete, hypers.py should also
be run as a job on the cluster.

To use hypers.py, modify hypers.job to update the hardcoded path within.  Launch hypers.job with: 

bsub < hypers.job

The main job will launch hypers.py, which itself will launch all the training jobs.
Each training job runs hypers.sh.  The hypers.sh file can contain anything you want, so long as it trains your PyTorch module and accepts as command-line parameters the following required values:
- learn_rate
- optim
- batch_size
- hidden_dim
- gid
- cid

The learn_rate should be a small float value, less than 1.0.  The optim value is the name of a PyTorch optimizer, such as 'Adam'.  The batch_size should be a modest-sized integer.
The cid and gid parameters are placeholders for hypers.py, which will provide the values when it generates the training job files.
The hidden_dim is a space-separated sequence of integers describing the dimensions of the network, one integer for each hidden layer specifying the width of that layer.  For example a 5x3x2 network would be: "--hidden_dim 5 3 2".

The training module must upon completion print the following lines to stdout, so that hypers.py can collect its results:

GID: [gid]

CID: [cid]

LOSS: [final validation loss value]

HOSTNAME: [hostname where the training job ran]

OK


That last "OK" indicates successful completion of the training job. See hypers.sh for an example of what you might put in the training script.
