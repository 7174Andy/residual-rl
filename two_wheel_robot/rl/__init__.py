"""RL utilities: wrappers, training entrypoints for the unicycle. The algorithm-agnostic
stable_baselines3 plumbing lives in `rl/sb3.py`; this subpackage still imports
stable_baselines3 directly for env construction (`Monitor`, `DummyVecEnv`)."""
