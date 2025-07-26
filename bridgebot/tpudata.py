# TODO: consider replacing module with tf.data.experimental.copy_to_device.
import tensorflow as tf
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.framework import function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import functional_ops
from tensorflow.python.training import server_lib


def ControllerDataset(source_ds_fn):
  worker_job = 'worker'
  source_job = 'coordinator'
  source_device = '/job:{}/task:0'.format(source_job)
  with ops.device(source_device):
    source_dataset = source_ds_fn()
    source_dataset_output_structure = dataset_ops.get_legacy_output_types(
        source_dataset)
    source_dataset = source_dataset.map(lambda *item: tf.nest.flatten(item))
    source_dataset_output_shapes = dataset_ops.get_legacy_output_shapes(
        source_dataset)

    source_iterator = dataset_ops.make_one_shot_iterator(source_dataset)
    source_handle = source_iterator.string_handle()


  @function.Defun(dtypes.string)
  def LoadingFunc(h):
    remote_iterator = iterator_ops.Iterator.from_string_handle(
        h, dataset_ops.get_legacy_output_types(source_dataset),
        dataset_ops.get_legacy_output_shapes(source_dataset))
    return remote_iterator.get_next()

  def MapFn(unused_input):
    source_dataset_output_types = dataset_ops.get_legacy_output_types(
        source_dataset)
    if isinstance(source_dataset_output_types, dtypes.DType):
      output_types = [source_dataset_output_types]
    elif isinstance(source_dataset_output_types, (list, tuple)):
      output_types = source_dataset_output_types
    else:
      raise ValueError('source dataset has invalid output types: {}'.format(source_dataset_output_types))
    remote_calls = functional_ops.remote_call(
        args=[source_handle],
        Tout=output_types,
        f=LoadingFunc,
        target='/job:%s/replica:0/task:0/cpu:0' % source_job)
    if len(remote_calls) == 1:
      return remote_calls[0]
    else:
      return remote_calls

  with ops.device('/job:%s' % worker_job):
    output_dataset = dataset_ops.Dataset.range(2).repeat().map(
        MapFn, num_parallel_calls=None)

    def reshape_and_unflatten(*item):
      reshapes = [[-1] + s.as_list()[1:] for s in source_dataset_output_shapes]
      reshapes = source_dataset_output_shapes
      item = [tf.reshape(v, s) for v, s in zip(item, reshapes)]
      return tf.nest.pack_sequence_as(source_dataset_output_structure, item)

    output_dataset = output_dataset.map(reshape_and_unflatten)
  return output_dataset
