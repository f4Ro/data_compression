"""
Entry point
"""
from benchmarking.conduct_experiments import conduct_full_experiment

# model names can be "rnn" or "cbn_vae"
# dataset names can be "intel" or "distillate_flow"

conduct_full_experiment('distillate_flow', 'cbn_vae', False, 3)
