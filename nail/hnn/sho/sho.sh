
# simple harmonic oscillator:
time python3  -u shomain.py \
  --hamiltonian="(p**2 + q**2) / 2" \
  --state_symbols q p --save_dir save \
  --name "sho-dataset-dsr1e-01-tspan0_100-traj2000" \
  --activation_fn Tanh --model hnn  --learn_rate 1e-03 \
  --batch_size 32 --hidden_dim  32 32 --train_pct 10 --test_pct 5 \
  --early_stop 9.9e-05


