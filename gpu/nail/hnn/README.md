This is my GPU code for the Hamiltonian Neural Network (HNN) which learns the trajectories of the N-Dimensional Simple Harmonic Oscillator (NDSHO).  The code runs in distributed-data parallel mode using PyTorch, yielding a very nice speedup over conventional CPU or single-GPU implementations.


To generate training data, use ndshogen.sh in ScottsML/nail/hnn/sho.  Edit the script first, to set the save_dir and name (output filename) parameters.


To run the experiments, first edit the sho/ndshox.sh script to set your save_dir and name (input filename) parameters, number of GPUs, etc.  Next add the gpu directory to your PYTHONPATH, and then run script.  For example:

export PYTHONPATH=/home/scottmiller/git/ScottsML/gpu:$PYTHONPATH
cd sho
./ndshox.sh --seed 42


By default the script uses 4 GPUs.  Run the script with only the --help parameter, to see the list of optional command-line parameters.

