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
"""Classes for supervised learning/training in Trax.

Trax provides classes for training supervised models:

  - NewSession: Run and record an n-step training session, with random
    initialization.

  - TrainTask: Labeled data + feedback mechanism (loss function w/ optimizer)
    for modifying a model's weights.

  - Optimizer: How to compute model weight updates using loss-derived gradients.
    May contain state ("slots", 1-1 with model weights) that accumulates across
    training steps. (This class is defined in the optimizers package.)

  - EvalTask: How and when to measure model performance as a function of
    training step number.
"""

from trax import layers as tl
from trax import math
from trax import shapes


class TrainingSession:
  """Session that trains a model via weight updates from a supervised task.

  A new training session randomly initializes a model and evolves its weights
  via feedback from a supervised task, batch by batch. A training session
  typically also runs periodic evals and saves some intermediate checkpoints.
  """

  def __init__(self, model, task, output_dir=None, evals=None,
               checkpoint_at=None):
    """Configures a `TrainingSession`.

    Args:
      model: Trax layer.
      task: TrainTask instance.
      output_dir: Path telling where to save outputs (evals and checkpoints).
          Can be None if both evals and checkpoint_at are None.
      evals: EvalTask instance or None. If None, don't do any evals.
      checkpoint_at: Function or None. Function (integer --> boolean) says,
          for step n, whether that step should have its checkpoint saved. If
          None, don't save any checkpoints.
    """
    self._model = model
    self._task = task
    self._output_dir = output_dir
    self._evals = evals
    self._checkpoint_at = _never if checkpoint_at is None else checkpoint_at
    self._eval_at = _never if evals is None else evals.eval_at
    self._step = None

  @property
  def step(self):
    """Returns current step number in this training session."""
    return self._step

  def run(self, n_steps=1):
    """Runs a training session for n steps, starting from random initialization.

    Args:
      n_steps: Stop training after completing n steps.
    """
    model, evals = self._model, self._evals
    weights, opt_slots, opt_params = self._init_model_and_optimizer()

    # The training loop.
    for step_i in range(1, n_steps + 1):
      self._step = step_i
      weights, opt_slots = self._run_one_step(weights, opt_slots, opt_params)
      if self._eval_at(step_i):
        evals.run(model, step_i)
      if self._checkpoint_at(step_i):  # pylint: disable=not-callable
        self._save_checkpoint()

  def _init_model_and_optimizer(self):
    """Initalizes the model (weights, states) and optimizer (slots, params)."""
    task = self._task
    weights, _ = self._model.init(task.input_signature)
    opt_slots, opt_params = task.optimizer.tree_init(weights)
    return weights, opt_slots, opt_params

  def _run_one_step(self, weights, opt_slots, opt_params):
    """Updates model weights by running one task/batch of training data."""
    model, task = self._model, self._task
    gradients = self._get_gradients(task)
    new_weights, new_opt_slots = task.optimizer.tree_update(
        self.step, gradients, weights, opt_slots, opt_params)
    model.weights = new_weights  # If omitted, causes JAX tracing error.
    return new_weights, new_opt_slots

  def _get_gradients(self, task):
    """Runs model on one batch to compute gradients: d(loss)/d(weights)."""
    model = self._model
    get_grads = math.grad(_loss_as_fn_of_weights)
    return get_grads(model.weights, model, task.loss_layer, task.next_batch())

  def _log_step(self, msg):
    """Logs message, labeled with the current training step number."""
    print(f'\nStep {self.step}: {msg}')

  def _save_checkpoint(self):
    """Saves checkpoint to disk for the current training step."""
    raise NotImplementedError


def _loss_as_fn_of_weights(weights, model, loss_layer, batch):
  """Defines model loss function whose 0'th arg is weights, for use by grad."""
  model.weights = weights
  compute_model_loss = tl.Serial(model, loss_layer)
  return compute_model_loss(batch)


class TrainTask:
  """A supervised task (labeled data + feedback mechanism) for training."""

  def __init__(self, labeled_data, loss_layer, optimizer):
    r"""Configures a training task.

    Args:
      labeled_data: Iterator of batches of labeled data tuples. Each tuple has
          1+ inputs (NumPy ndarrays) followed by an ndarray of target values.
      loss_layer: Layer that computes a scalar value (the "loss") by comparing
          model output $$\hat{y}=f(x)$$ to the target $$y$$.
      optimizer: Optimizer object that computes model weight updates from
          loss-function gradients.
    """
    self._labeled_data = labeled_data
    self._loss_layer = loss_layer
    self._optimizer = optimizer
    self._input_signature = shapes.signature(self._next_input())

  @property
  def loss_layer(self):
    return self._loss_layer

  @property
  def optimizer(self):
    return self._optimizer

  @property
  def input_signature(self):
    return self._input_signature

  def next_batch(self):
    """Returns one batch of labeled data: a tuple of input(s) plus label."""
    return next(self._labeled_data)

  def _next_input(self):
    inputs_plus_label = self.next_batch()
    inputs = inputs_plus_label[:-1]
    return inputs[0] if len(inputs) == 1 else inputs


class EvalTask:
  """Labeled data plus scalar functions for (periodically) measuring a model.

  An eval set specifies how (`labeled_data` + `metrics`) and when (`eval_at`)
  to measure a model as it is training. The variance of each scalar output is
  reduced by measuring over multiple (`eval_N`) batches and reporting the
  average from those measurements.
  """

  def __init__(self, labeled_data, metrics,
               names=None, eval_at=None, eval_N=10):
    r"""Configures a set of evals.

    Args:
      labeled_data: Iterator of batches of labeled data tuples. Each tuple has
          1+ inputs (NumPy ndarrays) followed by an ndarray of target values.
      metrics: List of layers; each computes a scalar value per batch by
          comparing model output $$\hat{y}=f(x)$$ to the target $$y$$.
      names: List of names, one for each item in `metrics`, in matching order,
          to be used when recording/reporting eval output. If None, generate
          default names: 'scalar_0', 'scalar_1', ...
      eval_at: Function (integer --> boolean) that says, for training step n,
          whether that step should run the evals. If None, run a single eval on
          step 1.
      eval_N: Integer N that specifies how many eval batches to run; the eval
          output is then the average of the scalar outputs from the N batches.
    """
    self._labeled_data = labeled_data
    self._metrics = metrics
    self._names = names or self._default_names()
    self._eval_at = eval_at or _step_1_only
    self._eval_N = eval_N  # pylint: disable=invalid-name
    self._check_init_values()

  @property
  def eval_at(self):
    return self._eval_at

  def run(self, model, step_n=None):
    """Runs and records all the scalar metrics in this eval set.

    Args:
      model: Layer, typically in the middle of a supervised training session.
      step_n: Current training step number for the model. Can be `None`, e.g.,
          for a model not currently being trained.
    """
    for name, scalar in zip(self._names, self._metrics):
      value = self._run_one_eval(model, scalar, self._eval_N)
      self._record(value, name, step_n)

  def _default_names(self):
    return [f'scalars_{i}' for i in range(len(self._metrics))]

  def _check_init_values(self):
    if len(self._metrics) != len(self._names):
      raise ValueError(f'number of metrics ({len(self._metrics)}) != '
                       f'number of names ({len(self._names)})')

  def _run_one_eval(self, model, scalar_layer, n_batches):
    """Runs one scalar N times (N batches) and computes its average value."""
    eval_fn = tl.Serial(model, scalar_layer)
    values = [eval_fn(self._next_batch()) for _ in range(n_batches)]
    return sum(values) / n_batches

  def _next_batch(self):
    """Returns one batch of labeled data: a tuple of input(s) plus label."""
    return next(self._labeled_data)

  def _record(self, value, name, step_n):
    print(f'Eval at step {step_n}: {name} = {value}')


def _never(step_n):
  """Returns False for all step numbers."""
  del step_n
  return False


def _step_1_only(step_n):
  """Returns true for step 1 only."""
  return step_n == 1
