
export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12
python /home/stmille3/src/basic/nail/hnn/pendulum/double/dphypers.py $@ \
  --dsr 0.01 \
  --early_stop 9e-10 \
  --epochs 16 \
  --hamiltonian="-(p1**2 + 2*p2**2 - 2*p1*p2*cos(q1 - q2) + (-3 + cos(2*(q1 - q2)))*(2*cos(q1) + cos(q2))) / (-3 + cos(2*(q1 - q2)))" \
  --hidden_dim 32 32 \
  --input_dim 6 \
  --save_dir '/home/stmille3/src/basic/nail/hnn/pendulum/double/save' \
  --name "dp-dataset-dsr1e-02-tspan0_100-traj125-xy-p1pi" \
  --test_pct 1 \
  --train_pct 0 \
  --train_pts 0 \
  --tspan 0 10  \
  --state_symbols 'q1' 'q2' 'p1' 'p2' 

