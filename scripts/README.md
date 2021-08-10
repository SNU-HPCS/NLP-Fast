# Analysis Tools

This directory consists of (1) runnable scripts and (2) parsers.
Also, we provide example analysis tools (written by jupyter) to analyze the profiled results.

## Runnable Scripts

For CPU and GPU models, you can use `run_all.sh` and `run_gpu_all.sh` scripts, respectively.
You can also manually configure the model parameters by modifying the `eval_type.conf` file.

## Parser

Similar to the runnable scripts, there are two parsers: `parse_all.py` and `parse_gpu_all.py` for CPU and GPU, respectively.
You can get the csv file (pandas dataframe) as the parsed result.

## Example analysis codes

We provide example jupyter scripts to describe how to use the parsed results.

First, please run the below codes defining various useful functions.
```python
import pandas as pd
import numpy as np

pd.options.display.max_rows = 10000
pd.options.display.max_columns = 5000


#############################################
########### Operation selection #############
#############################################
def manage_entry_ordering(_df, show_all=False):
    if (show_all):
        # Total order
        op_order = [
            "EMBED_LOOKUP", "EMBED_POSTPROCESSING",
            "ATTENTION_LAYER_MASK_CREATION", "ATTENTION_LAYER_Q_GEN", "ATTENTION_LAYER_K_GEN", 
            "ATTENTION_LAYER_SCORE_CAL", "ATTENTION_LAYER_SCORE_NORM", "ATTENTION_LAYER_MASK_SUB",
            "ATTENTION_LAYER_SOFTMAX", "ATTENTION_LAYER_V_GEN", "ATTENTION_LAYER_WEIGHTED_SUM",
            "ATTENTION_LAYER_HEAD_MERGE", "ATTENTION_FC", "ATTENTION_RESIDUAL_CONNECT", "ATTENTION_LAYERNORM",
            "FEEDFORWARD_PRE", "FEEDFORWARD_GELU", "FEEDFORWARD_POST", "FEEDFORWARD_RESIDUAL_CONNECT", "FEEDFORWARD_LAYERNORM"]
        return _df.reindex(columns=op_order, level=1)
    else:
        # Pinpointed targets
        op_total_targets = [
            "ATTENTION_LAYER_MASK_CREATION", "ATTENTION_LAYER_Q_GEN", "ATTENTION_LAYER_K_GEN",
            "ATTENTION_LAYER_SCORE_CAL", "ATTENTION_LAYER_SCORE_NORM", "ATTENTION_LAYER_MASK_SUB",
            "ATTENTION_LAYER_SOFTMAX", "ATTENTION_LAYER_V_GEN", "ATTENTION_LAYER_WEIGHTED_SUM",
            "ATTENTION_LAYER_HEAD_MERGE", "ATTENTION_FC", "ATTENTION_RESIDUAL_CONNECT", "ATTENTION_LAYERNORM",
            "FEEDFORWARD_PRE", "FEEDFORWARD_GELU", "FEEDFORWARD_POST", "FEEDFORWARD_RESIDUAL_CONNECT", "FEEDFORWARD_LAYERNORM"]

        op_pinpoint_targets = [
            "ATTENTION_LAYER_Q_GEN", "ATTENTION_LAYER_K_GEN", "ATTENTION_LAYER_SCORE_CAL",
            "ATTENTION_LAYER_SOFTMAX", "ATTENTION_LAYER_V_GEN", "ATTENTION_LAYER_WEIGHTED_SUM",
            "ATTENTION_FC", "FEEDFORWARD_PRE", "FEEDFORWARD_GELU", "FEEDFORWARD_POST"]

        df_stacked = _df.stack(0)
        df_stacked["OTHERS"] = 0.0
        for op in op_total_targets:
            if op not in op_pinpoint_targets:
                df_stacked["OTHERS"] += df_stacked[op]
        df_new = df_stacked.stack().unstack(-2).unstack(-1)
        
        op_order = op_pinpoint_targets + ["OTHERS"]
        return df_new.reindex(columns=op_order, level=1)

    
#############################################
########### Trim latency (0.0) ##############
#############################################
def trim_latencystat(_df):
    _df.loc[_df.latency == 0., 'latency'] = np.nan
    return _df

    
#############################################
########### Calculate Mem ratio #############
#############################################
def calculate_cachestat(_df):
    _df['CS_L3MPKI'] = (_df['CS_L3MISS'] / _df['CS_INST']) * 1000.
    _df['CS_L2MPKI'] = (_df['CS_L2MISS'] / _df['CS_INST']) * 1000.
    _df['CS_L1MPKI'] = (_df['CS_L1MISS'] / _df['CS_INST']) * 1000.
    return _df


#############################################
############### do pivot table ##############
#############################################
def do_pivot(df):
    df_lat_pivot = df.pivot_table(values=["latency", "CS_L3MPKI", "CS_L2MPKI", "CS_L1MPKI"],
                                  columns=["entry_name"],
                                  #index=["seq_len", "hidden_size", "ff_size", "lib_th", "hw_opt", "exec_mode", "batch_idx"],
                                  index=["seq_len", "hidden_size", "ff_size", "batch_th", "lib_th", "run_mode", "option"],
                                 )
    return df_lat_pivot
```

### Performance breakdown

You can get the performance breakdown with below code.
```python
##############################
### Perf. breakdown
##############################
# df = pd.read_csv("__parsed_csv_results_from_parser.py__")

def do_filter(_df):
    op_total_targets = [
        "ATTENTION_LAYER_MASK_CREATION", "ATTENTION_LAYER_Q_GEN", "ATTENTION_LAYER_K_GEN",
        "ATTENTION_LAYER_SCORE_CAL", "ATTENTION_LAYER_SCORE_NORM", "ATTENTION_LAYER_MASK_SUB",
        "ATTENTION_LAYER_SOFTMAX", "ATTENTION_LAYER_V_GEN", "ATTENTION_LAYER_WEIGHTED_SUM",
        "ATTENTION_LAYER_HEAD_MERGE", "ATTENTION_RESIDUAL_CONNECT", "ATTENTION_LAYERNORM"]
    
    df_filter = (_df['entry_name'] == op_total_targets[0])
    
    for _target in op_total_targets:
        _filter = (_df['entry_name'] == _target)
        df_filter = df_filter | _filter
    
    return _df[df_filter]
    
#######
### End-to-end latency
def cal_end_to_end_lat(_df):
    df_end2end = _df[(_df["batch_tid"] == 0) & ((_df["head_tid"] == 0) | (_df["head_tid"] == -1))].copy()
    df_end2end_lat = trim_latencystat(df_end2end)
    df_end2end_cache = calculate_cachestat(df_end2end_lat)
    df_end2end_pivot = df_end2end_cache.pivot_table(values=["latency"],
                                                    columns=["entry_name"],
                                                    index=["run_mode", "option", "seq_len", "hidden_size", "ff_size", "batch_th", "lib_th"],
                                                    aggfunc=sum
                                                   )
    df_end2end_pivot_ordered = manage_entry_ordering(df_end2end_pivot)
    return df_end2end_pivot_ordered

df_pivot = cal_end_to_end_lat(df)
display(df_pivot)
```

Then, you can get the parsed latency stats.

### Cache stat analysis

You can get the cache hit/miss information by using the below code.

```python
##############################
### Cache stat
##############################
df = pd.read_csv("__parsed_csv_file__")

def do_filter(_df):
    op_total_targets = [
        "ATTENTION_LAYER_MASK_CREATION", "ATTENTION_LAYER_Q_GEN", "ATTENTION_LAYER_K_GEN",
        "ATTENTION_LAYER_SCORE_CAL", "ATTENTION_LAYER_SCORE_NORM", "ATTENTION_LAYER_MASK_SUB",
        "ATTENTION_LAYER_SOFTMAX", "ATTENTION_LAYER_V_GEN", "ATTENTION_LAYER_WEIGHTED_SUM",
        "ATTENTION_LAYER_HEAD_MERGE", "ATTENTION_RESIDUAL_CONNECT", "ATTENTION_LAYERNORM"]
    
    df_filter = (_df['entry_name'] == op_total_targets[0])
    
    for _target in op_total_targets:
        _filter = (_df['entry_name'] == _target)
        df_filter = df_filter | _filter
    
    return _df[df_filter]
    
#######
### End-to-end latency
def cal_end_to_end_cachestat(_df):
    _df = _df[((_df["run_mode"] == "all_opt") & (_df["option"] == "prefetch")) |
               ((_df["run_mode"] == "baseline") & (_df["option"] == "none"))].copy()
    df_end2end_cache = calculate_cachestat(_df)
    df_end2end_pivot = df_end2end_cache.pivot_table(values=["CS_L3MPKI","CS_INST"],
                                                    columns=["entry_name"],
                                                    index=["seq_len", "hidden_size", "ff_size", "batch_th", "lib_th", "run_mode", "option"],
                                                    aggfunc=sum
                                                   )
    return df_end2end_pivot

df_pivot = cal_end_to_end_cachestat(df)
display(df_pivot)
```
