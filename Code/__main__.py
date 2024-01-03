from __future__ import print_function
import numpy as np
import random
import time
from getembeddings import GetEmbeddings
from args import parse_args
import itertools

# if __name__ == "__main__":
#     # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0:全部输出，1:只输出警告和错误，2:只输出错误

#     t1 = time.time()
#     num_of_seed = 1  # 32 original// 16 works well// 8 is not as good as 16
#     random.seed(num_of_seed)  
#     np.random.seed(num_of_seed)
#     # num_epochs = 10
#     # num_negative_samples = 1
#     GetEmbeddings(parse_args(), num_neg_samples=2, 
#                   num_epochs=10, counter=0, walk_length=80, window_size=5, pos_prob=0.3, neg_prob=0.3, number_walks=80)
#     t2 = time.time()
#     running_time = t2 - t1
#     if running_time < 900:
#         print(f'Running time: {running_time:.5f}s')
#     else:
#         print(f'Running time: {time.strftime("%H:%M:%S", time.gmtime(running_time))}')

if __name__ == "__main__":
    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # 0:全部输出，1:只输出警告和错误，2:只输出错误

    num_of_seed = [8] # 32 original// 16 works well// 8 is not as good as 16
    counter = 0
    for seed in num_of_seed:
        random.seed(seed)  
        np.random.seed(seed)

        walk_lengths = [80]
        window_sizes = 5
        num_walks = 80
        num_epochs = 10
        num_negative_samples = 2
        pos_probs = 0.3
        neg_probs = 0.3

        # walk_lengths = [60,80,100]
        # window_sizes = 5
        # num_walks = 80
        # num_epochs = 10
        # num_negative_samples = 2
        # pos_probs = 0.3
        # neg_probs = 0.3           

        method = 'HOCGE'
        results_filename = f'C:/Users/Lenovo/Desktop/MyCode/data/dat_dge/Results/{method}/wiki_jump.txt'
        # for size, sample, num_walk, length in itertools.product(window_sizes, num_negative_samples, num_walks, walk_length):
        # for pos_prob, neg_prob in itertools.product(pos_probs, neg_probs):
        for walk_length in walk_lengths:
        # for pos_prob in pos_probs:
            for i in range(2):
                t1 = time.time()
                # neg_probs = pos_prob        
                F1_score = GetEmbeddings(parse_args(), num_negative_samples, num_epochs, counter, 
                                walk_length, window_sizes, pos_probs, neg_probs, num_walks)
                # print(f'epoch: {epoch}, num_negative_sample: {num_negative_sample}')
                counter += 1
                t2 = time.time()        
                running_time = t2 - t1

                with open(results_filename, 'a') as f:
                    # f.write(f'num walk:{num_walk}, num_negative_sample: {sample},\n F1_score: {F1_score}, \nrunning_time: {time.strftime("%H:%M:%S", time.gmtime(running_time))}\n')
                    f.write(f'F1_score: {F1_score}, \nrunning_time: {time.strftime("%H:%M:%S", time.gmtime(running_time))}\n') 
                    f.write(f'counter:{counter}, seed:{seed}, walk length:{walk_length}\n')
                    f.write(f'pos_prob:{pos_probs}, neg_prob:{neg_probs}\n')
                    # f.write(f'window size:{size}, num_negative_sample: {sample}, ,num walk:{num_walk}, walk length:{length}\n F1_score: {F1_score}, \nrunning_time: {time.strftime("%H:%M:%S", time.gmtime(running_time))}\n')
                    f.write('---------------------------------------------\n')
                print(f'Running time: {time.strftime("%H:%M:%S", time.gmtime(running_time))}')

# python __main__.py --method deepWalk --input air --label-file T --directed --weighted --hon --clf-ratio 0.8 