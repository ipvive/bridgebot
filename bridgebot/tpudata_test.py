import tensorflow as tf

from tensorflow.core.protobuf import cluster_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.framework import function
from tensorflow.python.data.ops import readers
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.lib.io import python_io
from tensorflow.python.ops import functional_ops
from tensorflow.python.platform import test
from tensorflow.python.training import server_lib
from tensorflow.python.util import compat

import tpudata

_NUM_ITEMS = 11

class TPUDataTest(test.TestCase):
  def setUp(self):
    super(TPUDataTest, self).setUp()
    self._coord = server_lib.Server.create_local_server()
    self._worker = server_lib.Server.create_local_server()

    self._cluster_def = cluster_pb2.ClusterDef()
    worker_job = self._cluster_def.job.add()
    worker_job.name = 'worker'
    worker_job.tasks[0] = self._worker.target[len('grpc://'):]
    coord_job = self._cluster_def.job.add()
    coord_job.name = 'coordinator'
    coord_job.tasks[0] = self._coord.target[len('grpc://'):]

    session_config = config_pb2.ConfigProto(cluster_def=self._cluster_def)

    self._sess = session.Session(self._worker.target, config=session_config)
    self._worker_device = '/job:' + worker_job.name

  def testDataRange(self):
    dataset = tpudata.ControllerDataset(
            lambda: tf.data.Dataset.range(_NUM_ITEMS))
    with ops.device(self._worker_device):
      iterator = dataset_ops.make_initializable_iterator(dataset)
    self._sess.run(iterator.initializer)
    get_next = iterator.get_next()
    for i in range(_NUM_ITEMS):
      self.assertEqual(i, self._sess.run(get_next))

  def testPyRange(self):
    dataset = tpudata.ControllerDataset(lambda: tf.data.Dataset.from_generator(
      lambda: range(_NUM_ITEMS), output_types=tf.int32, output_shapes=[]))
    with ops.device(self._worker_device):
      iterator = dataset_ops.make_initializable_iterator(dataset)
    self._sess.run(iterator.initializer)
    get_next = iterator.get_next()
    for i in range(_NUM_ITEMS):
      self.assertEqual(i, self._sess.run(get_next))

  def testPyFeatures(self):
    def input_gen_fn():
      for n in range(_NUM_ITEMS):
        yield {"n": n}
    dataset = tpudata.ControllerDataset(lambda: tf.data.Dataset.from_generator(
      input_gen_fn, output_types={"n": tf.int32}, output_shapes={"n": []}))
    with ops.device(self._worker_device):
      iterator = dataset_ops.make_initializable_iterator(dataset)
    self._sess.run(iterator.initializer)
    get_next = iterator.get_next()
    for i in range(_NUM_ITEMS):
      self.assertEqual({"n": i}, self._sess.run(get_next))

if __name__ == '__main__':
  test.main()
