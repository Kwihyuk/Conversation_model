import random
import sys
import os
import time
import math




import numpy as np
import tensorflow as tf
import data_utils
# i don't know why below import needed

from six.moves import xrange




_buckets = [ (5,10), (10,15), (20,25), (30,40), (40,50) ] 


def read_data(max_size=None): 
	""" Read data from source and targt files and put into buckets.

	Args:

	it must be aligned with the source file : n-th line contains the desired output
	for n-th line from the source_path

	max_size : maximum number of lines to read, all other will be ignored;
		   if 0 or None, data files will be read completely ( no limit )

	Returns:
		data_set : a list of length len(_buckets); 
			   data_set[n] contains a list of (source, target) paris read from the
			   provided data filed that fit into the n-th bucket
			   i.e., such that len(source) < _buckets[n][0] and
			                   len(target) < _buckets[n][1] 
			         source and target are lists of token-ids.
        """

	print 'a'
	data_set = [  []  for _ in _buckets ]
	with tf.gfile.GFile('../data/Answer_token_ids.txt',mode="r") as A_file:
		with tf.gfile.GFile('../data/Question_token_ids.txt',mode="r") as Q_file:
			source, target = A_file.readline(), Q_file.readline()
			counter = 0
			while source and target and ( not max_size or counter < max_size):
				counter += 1 
				if counter % 10000 == 0:
					print(" reading data line %d" % ( counter ))
					sys.stdout.flush()
				source_ids = [int(x) for x in source.split()]
				target_ids = [int(x) for x in target.split()]
				target_ids.append(data_utils.EOS_ID)
				for bucket_id, (source_size, target_size) in enumerate(_buckets):
					if len(source_ids) < source_size and len(target_ids) < target_size:
						data_set[bucket_id].append([source_ids, target_ids])
						break
				source, target = A_file.readline(),Q_file.readline()
	return data_set

		





   


def main(unused_args):
	data_set = read_data()
	print data_set[0][0]

if __name__ == "__main__":
	tf.app.run()
