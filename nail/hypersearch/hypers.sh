
export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12
python /home/stmille3/src/basic/nail/hnn/pendulum/simple/sphypers.py $@ \
  --dsr 0.01 \
  --early_stop 9e-10 \
  --epochs 16 \
  --hamiltonian="(p**2)/2.0 + (1.0 - cos(q))" --state_symbols q p \
  --hidden_dim 32 32 \
  --input_dim 4 \
  --save_dir '/home/stmille3/src/basic/nail/hnn/pendulum/simple/save' \
  --name "sp-dataset-dsr1e-01-tspan0_100-traj1250" \
  --test_pct 1 \
  --train_pct 0 \
  --train_pts 0 \
  --tspan 0 10 

