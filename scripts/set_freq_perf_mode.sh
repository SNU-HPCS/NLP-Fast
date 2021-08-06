#!/bin/bash
for i in `seq 0 47`; do
    echo performance > /sys/devices/system/cpu/cpu${i}/cpufreq/scaling_governor
done
