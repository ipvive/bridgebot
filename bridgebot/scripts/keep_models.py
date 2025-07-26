from absl import flags, app
import logging
import os
import subprocess
import tempfile
import time

flags.DEFINE_string("model_dir", "gs://njt-serene-epsilon/models/sl/smp", "")

flags.DEFINE_integer("keep_models_steps", 10000, "")

FLAGS = flags.FLAGS




sleep_interval = 60

def current_models():
    names = subprocess.check_output(['gsutil', 'ls',
        os.path.join(FLAGS.model_dir, "model.ckpt*index")])
    names = names.decode('utf-8').split('\n')
    model_steps = [int(name.split("-")[-1].split('.')[0]) for name in names[:-1]]
    return model_steps


def keep_model(model_step):
    with tempfile.TemporaryDirectory() as td:
        tckpt = os.path.join(td, "checkpoint")
        with open(tckpt, "w") as f:
            print(f'model_checkpoint_path: "model.ckpt-{model_step}"', file=f)
            print(f'all_model_checkpoint_paths: "model.ckpt-{model_step}"', file=f)
        try:
            subprocess.check_call(['gsutil', '-m', 'cp',
                os.path.join(FLAGS.model_dir,
                    f"model.ckpt-{model_step}.*"), tckpt,
                os.path.join(FLAGS.model_dir, "keep", f"{model_step}")])
        except:
            logging.error(f"Failed to keep checkpoint {model_step}.")


def main(_):
    last_kept_step = -1
    while True:
        model_steps = current_models()
        for model_step in model_steps:
            if model_step % FLAGS.keep_models_steps == 0 and \
                    model_step > last_kept_step:
                keep_model(model_step)
                last_kept_step = model_step
        time.sleep(sleep_interval) 


if __name__ == "__main__":
    app.run(main)
