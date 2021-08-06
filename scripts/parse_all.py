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
	re_pat = "(?P<run_mode>\S+)\.(?P<option>\S+)" + \
		"_bth-(?P<batch_th>\d+)_lth-(?P<lib_th>\d+)_batch-(?P<batch_num>\d+)" + \
		"_head-(?P<head_num>\d+)_layer-(?P<layer_num>\d+)_seq-(?P<seq_len>\d+)" + \
		"_hsize-(?P<hidden_size>\d+)_ffsize-(?P<ff_size>\d+)"
	info_filter = re.compile(re_pat)
	res = info_filter.match(name).groupdict()

	print("fname: %s, parsed =>"%(name), res)

	res_dict = {
		'run_mode':res['run_mode'],
		'option':res['option'],
		'batch_th':int(res['batch_th']),
		'lib_th':int(res['lib_th']),
		'batch_num':int(res['batch_num']),
		'head_num':int(res['head_num']),
		'layer_num':int(res['layer_num']),
		'seq_len':int(res['seq_len']),
		'hidden_size':int(res['hidden_size']),
		'ff_size':int(res['ff_size']),
	}

	return res_dict

def parse_cachestat(fpath):
	# print("[DEBUG]", fpath)
	cachestat_dict = {}
	with open(fpath) as f:
		while True:
			batch_idx = tid = 0
			# 1. Find a base marker for CacheStat
			for l in f:
				if l.startswith('=====') and 'CacheStat' in l:
					re_pat = "===== CacheStat \(batch_idx: (?P<batch_idx>\d+), batch_tid: (?P<batch_tid>\d+), head_idx: (?P<head_idx>-?\d+), head_tid: (?P<head_tid>-?\d+)"
					filter = re.compile(re_pat)
					res = filter.match(l).groupdict()
					if ('batch_idx' not in res or 'batch_tid' not in res or 'head_idx' not in res or 'head_tid' not in res):
						failed("[Invalid format] fail to parse batch_idx, batch_tid, head_idx, head_tid in CacheStat (%s)"%(l))
					batch_idx = int(res['batch_idx'])
					batch_tid = int(res['batch_tid'])
					head_idx = int(res['head_idx'])
					head_tid = int(res['head_tid'])
					break
			else:
				break

			# extract all cache stats as many as possible
			# (NOTE), the first line is header
			header = next(f).split()
			res_dict = {h: [] for h in header}
			res_dict['entry_name'] = list()
			res_dict['layer_idx'] = list()

			for l in f:
				if l.startswith('===== CacheStat DONE ====='):
					break

				raw_entry_name, raw_data = [x.strip() for x in l.split(':')]
				if raw_entry_name.startswith('LAYER-'):
					layer_idx_start = len('LAYER-')
					layer_idx_end = layer_idx_start + raw_entry_name[layer_idx_start:].find('-')
					layer_idx = int(raw_entry_name[layer_idx_start:layer_idx_end])
					entry_name = raw_entry_name[layer_idx_end+1:]
				else:
					layer_idx = 0
					entry_name = raw_entry_name
				res_dict['entry_name'].append(entry_name)
				res_dict['layer_idx'].append(layer_idx)
				data = raw_data.strip().split()
				for i in range(0, len(data) // 2):
					runratio = min(100., float(data[i*2 + 1][1:-1])) / 100.
					if runratio == 0.:
						runratio = 1.
					value = int(data[i*2]) / runratio
					#print(fname, ",%s=%s" % (header[i+1], value), end='')
					res_dict[header[i]].append(value)

			cachestat_dict[(batch_idx, batch_tid, head_idx, head_tid)] = res_dict

	return cachestat_dict

def parse_latency(fpath):
	latency_dict = {}
	with open(fpath) as f:
		while True:
			batch_idx = tid = 0
			# 1. Find a base marker for CacheStat
			for l in f:
				if l.startswith('=====') and 'LatencyStat' in l:
					re_pat = "===== LatencyStat \(batch_idx: (?P<batch_idx>\d+), batch_tid: (?P<batch_tid>\d+), head_idx: (?P<head_idx>-?\d+), head_tid: (?P<head_tid>-?\d+)"
					filter = re.compile(re_pat)
					res = filter.match(l).groupdict()
					if ('batch_idx' not in res or 'batch_tid' not in res or 'head_idx' not in res or 'head_tid' not in res):
						failed("[Invalid format] fail to parse batch_idx, batch_tid, head_idx, head_tid in LatencyStat (%s)"%(l))
					batch_idx = int(res['batch_idx'])
					batch_tid = int(res['batch_tid'])
					head_idx = int(res['head_idx'])
					head_tid = int(res['head_tid'])
					break
			else:
				break

			# extract all cache stats as many as possible
			# (NOTE), the first line is header
			res_dict = {
				'entry_name':[],
				'latency':[],
				'layer_idx':[],
			}

			for l in f:
				if l.startswith('===== LatencyStat DONE ====='):
					break


				raw_entry_name, latency = [x.strip() for x in l.split(':')]
				if raw_entry_name.startswith('LAYER-'):
					layer_idx_start = len('LAYER-')
					layer_idx_end = layer_idx_start + raw_entry_name[layer_idx_start:].find('-')
					layer_idx = int(raw_entry_name[layer_idx_start:layer_idx_end])
					entry_name = raw_entry_name[layer_idx_end+1:]
				else:
					layer_idx = 0
					entry_name = raw_entry_name
				res_dict['entry_name'].append(entry_name)
				res_dict['layer_idx'].append(layer_idx)
				res_dict['latency'].append(float(latency))

			latency_dict[(batch_idx, batch_tid, head_idx, head_tid)] = res_dict

	return latency_dict

def main_parser(log_dname):
	global TEST_DIR

	test_info_dict = parse_execlogname(log_dname)
	filepath = os.path.join(TEST_DIR, log_dname, "log.txt")
	cachestats = parse_cachestat(filepath)
	latstats = parse_latency(filepath)
	if (len(cachestats) == 0 and len(latstats) == 0):
		print("No valid data in %s"%(filepath))
		return None

	# Sanity check
	if (len(cachestats) != len (latstats)):
		failed("# of entries should be same between latency & cache stats")
	if (test_info_dict["run_mode"] == "all_opt"):
		if (len(cachestats) != (test_info_dict['batch_num'] * (test_info_dict['head_num'] * 2 + 1))):
			failed("# of elements (%d) should be same with batch_num * (head_num + 1) (%d)"%(len(cachestats), test_info_dict['batch_num'] * (test_info_dict['head_num'] + 1)))
	else:
		if (len(cachestats) != (test_info_dict['batch_num'] * (test_info_dict['head_num'] + 1))):
			failed("# of elements (%d) should be same with batch_num * (head_num + 1) (%d)"%(len(cachestats), test_info_dict['batch_num'] * (test_info_dict['head_num'] + 1)))
	for (stat_key) in cachestats:
		cache_st = cachestats[stat_key]
		lat_st = latstats[stat_key]
		for entry_idx in range(len(cache_st['entry_name'])):
			if (cache_st['entry_name'][entry_idx] != lat_st['entry_name'][entry_idx]):
				failed("%dth entry name is different (lat: %s, cache: %s)"%(entry_idx, lat_st, cache_st))

	res_tot_df_list = []
	for stat_key in cachestats:
		(batch_idx, batch_tid, head_idx, head_tid) = stat_key
		cache_st = cachestats[stat_key]
		lat_st = latstats[stat_key]

		cache_df = pd.DataFrame.from_dict(cache_st)
		lat_df = pd.DataFrame.from_dict(lat_st)
		if (len(cache_df) != len(lat_df)):
			failed("corrupted dataframe len(cache:%d, lat:%d)"%(len(cache_df), len(lat_df)))
		# Merge two data frames by 'entry_name'
		tot_df = pd.merge(cache_df, lat_df, on=['entry_name', 'layer_idx'], how='outer')
		if (len(cache_df) != len(tot_df)):
			failed("Impossible case")

		new_column_dic = test_info_dict.copy()
		new_column_dic['batch_idx'] = batch_idx
		new_column_dic['batch_tid'] = batch_tid
		new_column_dic['head_idx'] = head_idx
		new_column_dic['head_tid'] = head_tid
		tot_df = tot_df.assign(**new_column_dic)
		res_tot_df_list.append(tot_df)

	return res_tot_df_list


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
