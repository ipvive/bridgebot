from absl import flags, app
import numpy as np
import re
import subprocess
import time

import pdb


flags.DEFINE_bool("debug", False, "show reason for failure")
flags.DEFINE_string("benchmarks", "",
        "comma-separated list of benchmarks to run")

FLAGS = flags.FLAGS

def mubert_bench_command(
        use_tpu=True,
        use_coordinator=True,
        config_name="tiny",
        batch_size=1024,
        num_actions=20,
        max_seq_length=256,
        hidden_state_length=32):
    config_file = "benchdata/in/bridgebot_config_{}.json".format(config_name)
    model_dir = "gs://ipvive-bot-benchmark-data/models/{}".format(config_name)
    return [
            "python3",
            "mubert_bench.py",
            "--bridgebot_config_file={}".format(config_file),
            "--use_tpu={}".format(use_tpu),
            "--use_coordinator={}".format(use_coordinator),
            "--tpu_name=benchmark",
            "--num_examples=8192",
            "--num_actions={}".format(num_actions),
            "--model_dir={}".format(model_dir),
            "--batch_size={}".format(batch_size),
            "--max_seq_length={}".format(max_seq_length),
            "--hidden_state_length={}".format(hidden_state_length),
    ]


benchmark_commands = {}
for n in ["small", "tiny"]:
    benchmark_commands.update({
        "{}-default".format(n): mubert_bench_command(config_name=n),
        "{}-no_tpu".format(n): mubert_bench_command(config_name=n, use_tpu=False, use_coordinator=False),
        "{}-no_coord".format(n): mubert_bench_command(config_name=n, use_coordinator=False),
        "{}-batch=128".format(n): mubert_bench_command(config_name=n, batch_size=128),
        "{}-short_seq".format(n): mubert_bench_command(config_name=n, max_seq_length=64),
        "{}-long_hidden".format(n): mubert_bench_command(config_name=n, hidden_state_length=64),
        "{}-many_actions".format(n): mubert_bench_command(config_name=n, num_actions=100),
    })


def capture_benchmark(dt, commit, name, cmd):
    if FLAGS.debug:
        print("running {}".format(" ".join(cmd)))
    result = subprocess.run(cmd, stdout=subprocess.PIPE)
    times = []
    last_secs = None
    for line in result.stdout.decode("utf8").split("\n"):
        m = re.match(r"([\d\.]+) infer batch ready.*", line)
        if m:
            secs = float(m.group(1))
            if last_secs is not None:
                times.append(secs - last_secs)
            last_secs = secs
    if len(times) < 2:
        return "{} {} {} failed".format(dt, commit, name)
    else:
        times = np.array(times[1:])
        return "{} {} {} sum={} mean={} std={} raw={}".format(dt, commit, name, np.sum(times), np.mean(times), np.std(times), list(times))


def log_lines(logpath, lines):
    with open(logpath, "a") as f:
        for l in lines:
            print(l, file=f)


def capture_benchmarks(_):
    if FLAGS.benchmarks:
        benchmarks = FLAGS.benchmarks.split(",")
    else:
        benchmarks = benchmark_commands.keys()
        benchmarks = [b for b in benchmarks if b != "small-no_tpu"]
    t = time.localtime()
    datestr = time.strftime("%Y%m%d", t)
    dt = time.strftime("%Y%m%d %H:%M:%S", t)
    headerlines = []
    logpath = "benchdata/out/{}.log".format(datestr)
    res = subprocess.run(["git", "log", "-n", "1"], stdout=subprocess.PIPE)
    headerlines.append("# ===================================")
    headerlines.extend(["#" + l for l in res.stdout.decode("utf8").split("\n")])
    commit = res.stdout.decode("utf8").split(" ")[1][:6]
    res = subprocess.run(["git", "status", "--porcelain"], stdout=subprocess.PIPE)
    if res.stdout != "":
        headerlines.extend(["#" + l for l in res.stdout.decode("utf8").split("\n")])
    headerlines.append("# ===================================")
    log_lines(logpath, headerlines)
    for name in benchmarks:
        line = capture_benchmark(dt, commit, name, benchmark_commands[name])
        log_lines(logpath, [line])


if __name__ == "__main__":
    app.run(capture_benchmarks)
