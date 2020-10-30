
# Run one experiment for simple pendulum.  Change the random seed using the --seed arg, e.g.:
# ./spx.sh --seed 42

# threading hints for the cluster:
export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12

python spx.py $@ \
  --activation_fn 'Tanh' \
  --batch_size 1 \
  --clip 0 \
  --dsr 0.01 \
  --early_stop 9e-10 \
  --epochs 16 \
  --eps 1e-08 \
  --hamiltonian="(p**2)/2.0 + (1.0 - cos(q))" --state_symbols q p \
  --hidden_dim 32 32 \
  --input_dim 4 \
  --input_noise '' \
  --integrator_scheme 'RK45' \
  --learn_rate 0.001 \
  --load_model '' \
  --model 'baseline' \
  --momentum 0 \
  --optim 'Adam' \
  --print_every 200 \
  --save_dir save \
  --name "spdata" \
  --test_pct 20 \
  --total_steps 4000 \
  --train_pct 0 \
  --train_pts 0 \
  --trajectories 1000 \
  --tspan 0 10 \
  --weight_decay 0  
