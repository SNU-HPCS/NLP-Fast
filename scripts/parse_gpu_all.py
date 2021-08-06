#!/usr/bin/python3
from collections import defaultdict
import argparse
import os
import re
import pandas as pd
import multiprocessing

def failed(why):
	print("[FAILED] ", why)
	exit(1)

def parse_execlogname(name):
	re_pat = "(?P<eval_type>\S+)-gpu_stream-(?P<cuda_stream>\d+)_tb-(?P<thread_block>\d+)_batch-(?P<batch_num>\d+)" + \
			 "_head-(?P<head_num>\d+)_layer-(?P<layer_num>\d+)_seq-(?P<seq_len>\d+)" + \
			 "_hsize-(?P<hidden_size>\d+)_ffsize-(?P<ff_size>\d+)_iter-(?P<iter>\d+)"
	info_filter = re.compile(re_pat)
	res = info_filter.match(name).groupdict()

	print("fname: %s, parsed =>"%(name), res)

	res_dict = {
		'eval_type':res['eval_type'],
		'cuda_stream':int(res['cuda_stream']),
		'thread_block':int(res['thread_block']),
		'batch_num':int(res['batch_num']),
		'head_num':int(res['head_num']),
		'layer_num':int(res['layer_num']),
		'seq_len':int(res['seq_len']),
		'hidden_size':int(res['hidden_size']),
		'ff_size':int(res['ff_size']),
		'iter':int(res['iter']),
	}

	return res_dict


def parse_latency(fpath):
	with open(fpath) as f:
		for l in f:
			re_pat = "elapsed: (?P<latency>\d+\.\d+)"
			filter = re.compile(re_pat)
			m_obj = filter.match(l)
			if m_obj is not None:
				res = m_obj.groupdict()
				return float(res['latency'])
	return 0.0

def main_parser(log_dname):
	global TEST_DIR

	test_info_dict = parse_execlogname(log_dname)
	filepath = os.path.join(TEST_DIR, log_dname, "log.txt")
	latency = parse_latency(filepath)

	new_dic = defaultdict(list)
	for key in test_info_dict:
		new_dic[key] = [test_info_dict[key]]
	new_dic['latency'] = [latency]
	return [pd.DataFrame.from_dict(new_dic)]

if __name__ == '__main__':
	global TEST_DIR

	parser = argparse.ArgumentParser()
	parser.add_argument('--test_dir', help='a directory including experimental results for specific testcase (MUST be given)')
	parser.add_argument('--csv_dir', help='a directory where result csv file is created (default: ./)')
	ARGS = parser.parse_args()

	if ((ARGS.test_dir == None) or
			(not os.path.isdir(ARGS.test_dir))):
		print("Invalid [test_dir]. Check -h")
		exit(1)
	TEST_DIR = ARGS.test_dir
	if (ARGS.csv_dir == None):
		CSV_DIR = "./"
	elif (os.path.isdir(ARGS.csv_dir)):
		CSV_DIR = ARGS.csv_dir
	else:
		print("Invalid [csv_dir]. check -h")
		exit(1)

	OUTPUT_CSV_FNAME = "%s/%s"%(CSV_DIR, TEST_DIR.rstrip('/').split('/')[-1] + '_df.csv')

	tot_df_list = []
	with multiprocessing.Pool(50) as p:
		res_df_list_pool = p.map(main_parser, os.listdir(TEST_DIR))

	for _res_df_list in res_df_list_pool:
		if (_res_df_list == None):
			continue
		tot_df_list += _res_df_list

	final_df = pd.concat(tot_df_list)
	# Save the processed dataframe in csv format
	print("Save csv to", OUTPUT_CSV_FNAME)
	final_df.to_csv(OUTPUT_CSV_FNAME, index=False)
