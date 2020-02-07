# coding=utf-8
# Copyright 2020 The Trax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Test fully training an MNIST model (2000 steps)."""

from absl.testing import absltest

import gin

from trax import layers as tl
from trax.optimizers import adafactor
from trax.supervised import inputs
from trax.supervised import training


class MnistTest(absltest.TestCase):

  def test_train_mnist(self):
    """Train MNIST model fully, to compare against other implementations.

    Evals for cross-entropy loss and accuracy are run every 100 steps; their
    values are visible in the test log.
    """
    gin.parse_config([
        'batch_fn.batch_size_per_device = 256',
        'batch_fn.eval_batch_size = 256',
    ])

    mnist_model = tl.Serial(
        tl.Flatten(),
        tl.Dense(512),
        tl.Relu(),
        tl.Dense(512),
        tl.Relu(),
        tl.Dense(10),
        tl.LogSoftmax(),
    )
    task = training.TrainTask(_training_data(),
                              tl.CrossEntropyLoss(),
                              adafactor.Adafactor(.02))
    evals = training.EvalTask(_eval_data(),
                              [tl.CrossEntropyLoss(), tl.AccuracyScalar()],
                              names=['CrossEntropyLoss', 'AccuracyScalar'],
                              eval_at=_every_nth_step(100),
                              eval_N=10)

    session = training.TrainingSession(mnist_model, task, evals=evals)
    session.run(n_steps=2000)
    self.assertEqual(session.step, 2000)


def _mnist_dataset():
  """Loads (and caches) the standard MNIST data set."""
  return inputs.inputs('mnist')


def _training_data():
  """Cycles through the MNIST training data as many times as needed."""
  while True:
    for batch in _mnist_dataset().train_stream(1):
      yield batch


def _eval_data():
  """Cycles through the MNIST eval data as many times as needed."""
  while True:
    for batch in _mnist_dataset().eval_stream(1):
      yield batch


def _every_nth_step(n):
  """Returns a function that returns True only on every n'th training step."""
  return lambda step_n: step_n % n == 0


if __name__ == '__main__':
  absltest.main()
