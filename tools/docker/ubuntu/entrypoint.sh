set -e
pip install -e /pyrltools\[mkl\]
python3 /pyrltools/examples/pendulum_sac.py