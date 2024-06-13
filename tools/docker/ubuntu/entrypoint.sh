set -e
pip install -e /rltools\[mkl\]
python3 /rltools/examples/pendulum_sac.py