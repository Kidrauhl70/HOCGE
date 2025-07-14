from __future__ import print_function
import numpy as np
import random
import time
import os
from getembeddings import GetEmbeddings
from args import parse_args

if __name__ == "__main__":
    # parameters
    seed = 42
    walk_length = 80
    window_size = 5
    num_walks = 80
    num_epochs = 20
    num_negative_samples = 3
    pos_prob = 0.3
    neg_prob = 0.3
    method = 'HOCGE'
    counter = 0

    # Set random seed
    random.seed(seed)
    np.random.seed(seed)

    # Output path
    results_dir = os.path.join("results", method)
    os.makedirs(results_dir, exist_ok=True)
    results_filename = os.path.join(results_dir, "tube-1106-2.txt")

    # Run embedding and evaluation
    t1 = time.time()
    F1_score = GetEmbeddings(
        args=parse_args(),
        num_neg_samples=num_negative_samples,
        num_epochs=num_epochs,
        counter=counter,
        walk_length=walk_length,
        window_size=window_size,
        pos_prob=pos_prob,
        neg_prob=neg_prob,
        number_walks=num_walks
    )
    t2 = time.time()
    runtime = time.strftime("%H:%M:%S", time.gmtime(t2 - t1))

    # Write result
    with open(results_filename, 'a') as f:
        f.write(f'pos: {pos_prob}, neg: {neg_prob}, num_sample: {num_negative_samples}\n')
        f.write(f'num_negative_sample: {num_negative_samples}, num_walk: {num_walks}, walk_length: {walk_length}\n')
        f.write(f'F1_score: {F1_score}, running_time: {runtime}, seed: {seed}, counter: {counter}\n')
        f.write('---------------------------------------------\n')

    print(f'Running time: {runtime}')
