#!/usr/bin/env python

import sys
import os
from timeit import default_timer as timer



if __name__ == "__main__":

    if len(sys.argv) != 2:
        sys.exit('Error: expected path to directory')

    dir_path = sys.argv[1]
    files = os.listdir(dir_path)

    word_count_dict = {}
    separator = '\t'

    start = timer()

    for filename in files:
        try:
            with open(os.path.join(dir_path, filename), encoding="utf-8", errors='ignore') as f:

                for line in f:
                    words = line.rstrip().split()
                    
                    for w in words:
                        if w in word_count_dict:
                            word_count_dict[w] += 1
                        else:
                            word_count_dict[w] = 1
        except IOError as error:
           sys.exit(f'Error: Unable to read file {filename} in directory {dir_path}') 

    with open('output.txt', 'w') as f:
        for key, value in word_count_dict.items():
            text = key.encode('ascii', 'ignore')
            print(f'{text}{separator}{value}', file=f)

    end = timer()
    print(end - start)

