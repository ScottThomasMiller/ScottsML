This is my GPU code for the Hamiltonian Neural Network (HNN) which learns the trajectories of the N-Dimensional Simple Harmonic Oscillator (NDSHO).  The code runs in distributed-data parallel mode using PyTorch, yielding a very nice speedup over conventional CPU or single-GPU implementations.

To generate training data, use ndshogen.sh in ScottsML/nail/hnn/sho.  Edit the script first, to set the save_dir and name (output filename) parameters.

To run the experiments, first edit the sho/ndshox.sh script to set your save_dir and name (input filename) parameters, number of GPUs, etc.  Next add the gpu directory to your PYTHONPATH, and then run script.  For example:

export PYTHONPATH=/home/scottmiller/git/ScottsML/gpu:$PYTHONPATH
cd sho
./ndshox.sh --seed 42

By default the script uses 4 GPUs.  Run the script with only the --help parameter, to see the following list of optional command-line parameters:

usage: ndshox.py [-h] [--hamiltonian HAMILTONIAN] [--clip CLIP]
                 [--num_bodies NUM_BODIES] [--num_gpus NUM_GPUS]
                 [--epochs EPOCHS] [--input_dim INPUT_DIM]
                 [--hidden_dim [HIDDEN_DIM [HIDDEN_DIM ...]]]
                 [--momentum MOMENTUM] [--learn_rate LEARN_RATE] [--eps EPS]
                 [--weight_decay WEIGHT_DECAY] [--batch_size BATCH_SIZE]
                 [--train_pct TRAIN_PCT] [--train_pts TRAIN_PTS]
                 [--master_port MASTER_PORT] [--optim OPTIM]
                 [--input_noise INPUT_NOISE] [--print_every PRINT_EVERY]
                 [--integrator_scheme INTEGRATOR_SCHEME]
                 [--activation_fn ACTIVATION_FN] [--name NAME] [--model MODEL]
                 [--early_stop EARLY_STOP] [--energy ENERGY]
                 [--split_ratio SPLIT_RATIO] [--dsr DSR] [--save_stats]
                 [--verbose] [--save_dir SAVE_DIR] [--load_model LOAD_MODEL]
                 [--test_pct TEST_PCT] [--gpu GPU]
                 [--trajectories TRAJECTORIES] [--seed SEED]
                 [--total_steps TOTAL_STEPS] [--tspan [TSPAN [TSPAN ...]]]
                 [--state_symbols STATE_SYMBOLS [STATE_SYMBOLS ...]]

optional arguments:
  -h, --help            show this help message and exit
  --hamiltonian HAMILTONIAN
                        Hamiltonian of the system
  --clip CLIP           gradient clipping
  --num_bodies NUM_BODIES
                        number of bodies/particles in the system
  --num_gpus NUM_GPUS   number of GPUs
  --epochs EPOCHS       number of training epochs
  --input_dim INPUT_DIM
                        Input dimension
  --hidden_dim [HIDDEN_DIM [HIDDEN_DIM ...]]
                        hidden layers dimension
  --momentum MOMENTUM   momentum for SGD
  --learn_rate LEARN_RATE
                        learning rate
  --eps EPS             epsilon for numerical stability
  --weight_decay WEIGHT_DECAY
                        weight decay
  --batch_size BATCH_SIZE
                        batch size
  --train_pct TRAIN_PCT
                        use pct of total available training data
  --train_pts TRAIN_PTS
                        use number of total available training data
  --master_port MASTER_PORT
                        TCP port number for master process
  --optim OPTIM         optimizer
  --input_noise INPUT_NOISE
                        noise strength added to the inputs
  --print_every PRINT_EVERY
                        printing interval
  --integrator_scheme INTEGRATOR_SCHEME
                        name of the integration scheme [RK4, RK45, Symplectic]
  --activation_fn ACTIVATION_FN
                        which activation function to use [Tanh, ReLU]
  --name NAME           Name of the system
  --model MODEL         baseline or hamiltonian
  --early_stop EARLY_STOP
                        validation loss value for early stopping
  --energy ENERGY       fixed energy level
  --split_ratio SPLIT_RATIO
                        test data split ratio
  --dsr DSR             data sampling rate
  --save_stats          save stats. USES DOUBLE MEMORY
  --verbose             Verbose output or not
  --save_dir SAVE_DIR   where to save the trained model
  --load_model LOAD_MODEL
                        name of the previously saved model to resume
  --test_pct TEST_PCT   percentage of total train data to use for test set
  --gpu GPU             target GPU device #
  --trajectories TRAJECTORIES
                        number of trajectories to generate
  --seed SEED           random generator seed
  --total_steps TOTAL_STEPS
                        No. of steps for gradient descent
  --tspan [TSPAN [TSPAN ...]]
                        timespan
  --state_symbols STATE_SYMBOLS [STATE_SYMBOLS ...]
                        state symbols
                        
