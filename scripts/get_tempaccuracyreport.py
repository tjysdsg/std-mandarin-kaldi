
#!/usr/bin/env python

# Copyright 2016    Vijayaditya Peddinti.
#           2016    Vimal Manohar
# Apache 2.0.

""" This script is based on steps/nnet3/chain/train.sh
"""

import argparse
import logging
import os
import pprint
import shutil
import sys
import traceback
import re
import datetime

sys.path.insert(0, 'steps')
import libs.nnet3.train.common as common_train_lib
import libs.common as common_lib
import libs.nnet3.train.chain_objf.acoustic_model as chain_lib
import libs.nnet3.report.log_parse as nnet3_log_parse


logger = logging.getLogger('libs')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(pathname)s:%(lineno)s - "
                              "%(funcName)s - %(levelname)s ] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.info('Starting chain model trainer (train.py)')

dir=sys.argv[1]

def parse_prob_logs(exp_dir, key='accuracy', output="output"):
    train_prob_files = "%s/log/compute_prob_train.*.log" % (exp_dir)
    valid_prob_files = "%s/log/compute_prob_valid.*.log" % (exp_dir)
    train_prob_strings = common_lib.get_command_stdout(
        'grep -e {0} {1}'.format(key, train_prob_files))
    valid_prob_strings = common_lib.get_command_stdout(
        'grep -e {0} {1}'.format(key, valid_prob_files))

    parse_regex = re.compile(
        ".*compute_prob_.*\.([0-9]+).log:LOG "
        ".nnet3.*compute-prob.*:PrintTotalStats..:"
        "nnet.*diagnostics.cc:[0-9]+. Overall ([a-zA-Z\-]+) for "
        "'{output}'.*is ([0-9.\-e]+) .*per frame".format(output=output))

    train_objf = {}
    valid_objf = {}

    for line in train_prob_strings.split('\n'):
        mat_obj = parse_regex.search(line)
        if mat_obj is not None:
            groups = mat_obj.groups()
            if groups[1] == key:
                train_objf[int(groups[0])] = groups[2]
    if not train_objf:
        raise KaldiLogParseException("Could not find any lines with {k} in "
                " {l}".format(k=key, l=train_prob_files))

    for line in valid_prob_strings.split('\n'):
        mat_obj = parse_regex.search(line)
        if mat_obj is not None:
            groups = mat_obj.groups()
            if groups[1] == key:
                valid_objf[int(groups[0])] = groups[2]

    if not valid_objf:
        raise KaldiLogParseException("Could not find any lines with {k} in "
                " {l}".format(k=key, l=valid_prob_files))

    iters = list(set(valid_objf.keys()).intersection(list(train_objf.keys())))
    if not iters:
        raise KaldiLogParseException("Could not any common iterations with"
                " key {k} in both {tl} and {vl}".format(
                    k=key, tl=train_prob_files, vl=valid_prob_files))
    iters.sort()
    return list([(int(x), float(train_objf[x]),
                               float(valid_objf[x])) for x in iters])


def generate_acc_logprob_report(exp_dir, key="accuracy", output="output"):
    import traceback
    try:
        times = nnet3_log_parse.get_train_times(exp_dir)
    except:
        tb = traceback.format_exc()
        logger.warning("Error getting info from logs, exception was: " + tb)
        times = {}

    report = []
    report.append("%Iter\tduration\ttrain_objective\tvalid_objective\tdifference")
    try:
        if key == "rnnlm_objective":
            data = list(nnet3_log_parse.parse_rnnlm_prob_logs(exp_dir, 'objf'))
        else:
            data = list(parse_prob_logs(exp_dir, key, output))
    except:
        tb = traceback.format_exc()
        nnet3_log_parse.logger.warning("Error getting info from logs, exception was: " + tb)
        data = []
    for x in data:
        try:
            report.append("%d\t%s\t%g\t%g\t%g" % (x[0], str(times[x[0]]), x[1], x[2], x[2]-x[1]))
        except (KeyError, IndexError):
            continue

    total_time = 0
    for iter in times.keys():
        total_time += times[iter]
    report.append("Total training time is {0}\n".format(str(datetime.timedelta(seconds=total_time))))
    return ["\n".join(report), times, data]

[report, times, data] = generate_acc_logprob_report(dir)
with open("{dir}/new_accuracy.report".format(dir=dir), "w") as f:
    f.write(report)


