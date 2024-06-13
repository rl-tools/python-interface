# Embedding Checkpoints into C/C++ Code

This example shows how checkpoints exported from the training scripts for (PPO, SAC, and TD3) can be embedded in C/C++ code (e.g. in a firmware).

To build and execute run e.g.:
```
clang++ -std=c++17 test.cpp -I../../rltools/external/rl_tools/include && ./a.out
```