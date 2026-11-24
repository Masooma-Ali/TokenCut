"""ODE: the decision engine that picks a model and trim level (an "arm") per task.

A contextual bandit (hybrid LinUCB, Li et al. 2010): at each new task it reads a short list of
numbers describing the task (x), scores every allowed arm, picks one, and learns from that
arm's reward when the task ends. Phase 1 is the bandit alone (`bandit.py`); features, arms,
reward, storage and the gateway policy come in later phases (see the ODE review doc).
"""
