import pennylane as qml
import pennylane.numpy as np
import matplotlib.pyplot as plt
import dill
import json
import datetime
import logging
import copy
import time
import os.path as opath

from generator import Generator

MAX_SIDE = 4


class Model(Generator):
	steps = 100
	objective = 1e-5
	max_steps = 100
	device = None
	circuit = None
	model = lambda circ, input, params: circ(input, params)  # noqa: E731
	params = None
	optimizer = qml.AdamOptimizer(stepsize=0.1)
	n_wires = 1

	batches = 1
	name = None
	logger = None
	logname = None
	logging = True
	checkpoints = []

	losses = []
	extra = {}

	current_time_step = 0
	time_per_iteration = 0
	time_estimation = 0

	def __init__(self, name, log=True, seed=None, path=None):
		super().__init__()

		self.name = name

		if seed is not None:
			self.rng = np.random.default_rng(seed)

		self.logging = log

		if log and (path is None):
			self.logger = logging.getLogger(__name__)
			self.logname = (
				f'logs/{name}{datetime.datetime.today().strftime("_%d-%m-%Y_%H:%M:%S")}.log'
			)
			logging.basicConfig(
				filename=self.logname,
				level=logging.INFO,
				format='[%(asctime)s] %(levelname)s %(message)s',
				datefmt='%H:%M:%S',
			)
		else:
			self.logging = False

		if path is not None:
			if opath.exists(path):
				file = open(path, 'r')
				bundle = file.read(-1)

				obj = json.loads(bundle)
			else:
				obj = json.loads(path)

			self.params = obj['params']
			self.n_wires = obj['n_wires']
			self.SAMPLE_COUNT = obj['sample_count']

			self.generate_device(self.n_wires)

			self.set_steps(obj['steps'])
			self.extra = dill.loads(bytes(obj['extra']))
			self.set_circuit(dill.loads(bytes(obj['circuit'])))
			self.set_model(dill.loads(bytes(obj['model'])))
			self.set_optimizer(dill.loads(bytes(obj['optimizer'])))
			self.set_function(dill.loads(bytes(obj['func'])))
			self.set_sample_size(obj['sample_size'])
			self.set_consts(obj['consts'])

			for item in obj['scalars'].values():
				if item.get('type') is not None:
					item['type'] = dill.loads(bytes(item['type']))
			self.set_scalars(obj['scalars'])

			self.set_angles(obj['angles'])
			self.set_percent(obj['percent'])
			self.set_name(obj['name'])

			self.losses = obj['losses']
			self.checkpoints = dill.loads(bytes(obj['checkpoints']))

			try:
				self.read_samples('samples/' + self.name + '.bin')
			except FileNotFoundError:
				self.generate_samples()
				self.read_samples('samples/' + self.name + '.bin')

			if obj.get('train_in') is None:
				self.separate()
			else:
				self.train_in = dill.loads(bytes(obj['train_in']))
				self.train_out = dill.loads(bytes(obj['train_out']))
				self.test_in = dill.loads(bytes(obj['test_in']))
				self.test_out = dill.loads(bytes(obj['test_out']))

		return

	def generate_device(self, n_wires):
		self.n_wires = n_wires

		self.device = qml.device('lightning.qubit', wires=n_wires)

		if self.logging:
			self.logger.info(f'Created device with {self.n_wires} wires.')

		return

	def set_optimizer(self, optimizer):
		self.optimizer = optimizer

		if self.logging:
			self.logger.info('Set optimizer.')

		return

	def set_batches(self, batches):
		self.batches = batches

		if self.logging:
			self.logger.info('Set batches.')

		return

	def reset_optimizer(self):
		self.optimizer.reset()

		if self.logging:
			self.logger.info('Reset optimizer.')

		return

	def push_extra(self, name: str, data):
		self.extra[name] = data

		return

	def rollback(self, x):
		if x < len(self.checkpoints):
			self.params = copy.deepcopy(self.checkpoints[x].get('params'))
		else:
			raise ValueError(
				f'Index too large. Trying index {x} in list with {len(self.checkpoints)} elements.'
			)

		return

	def set_circuit(self, circuit):
		self._circuit = circuit
		self.circuit = qml.QNode(circuit, self.device)

		if self.logging:
			self.logger.info('Set circuit.')

		self.reset_optimizer()

		return

	def set_steps(self, steps):
		self.steps = steps

		if self.logging:
			self.logger.info('Set number of steps.')

		return

	def set_objective(self, objective):
		self.objective = objective

		if self.logging:
			self.logger.info('Set objective.')

		return

	def set_max_steps(self, max_steps):
		self.max_steps = max_steps

		if self.logging:
			self.logger.info('Set maximum number of steps.')

		return

	def set_model(self, model):
		self.model = model

		if self.logging:
			self.logger.info('Set model.')

		return

	def generate_params(self, shape):
		self.params = self.rng.random(shape, requires_grad=True) * np.pi * 2 - np.pi

		if self.logging:
			self.logger.info('Generated parameters.')

		return

	def loss(self, params, input, output):
		predictions = np.array(
			[self.model(self.circuit, single, params) for single in input]
		).flatten()

		loss = np.mean((output - predictions) ** 2)

		return loss

	def callback_text(self, step, val):
		print('Step', step, '-> Loss:', val, flush=True)

		return

	def callback_graph(self, last_time):
		from IPython.display import clear_output

		clear_output(wait=True)

		plt.plot(range(1, len(self.losses) + 1), self.losses)

		plt.xlabel('Step')
		plt.ylabel('Loss')

		plt.yscale('log')

		plt.grid(True)

		plt.show()

		now = time.time_ns()
		elapsed = int((now - last_time) / 1_000_000)

		self.time_per_iteration *= self.current_time_step
		self.current_time_step += 1
		self.time_per_iteration += elapsed
		self.time_per_iteration = self.time_per_iteration / self.current_time_step
		self.time_estimation = (self.max_steps - self.current_time_step) * self.time_per_iteration

		tmp = int(self.time_estimation)

		milis = tmp % 1000
		tmp = tmp // 1000
		secs = tmp % 60
		tmp = tmp // 60
		mins = tmp % 60
		tmp = tmp // 60
		hours = tmp

		print('Loss:', self.losses[-1])
		print('Step', len(self.losses))
		print(
			f'Estimated time remaining: {hours:02}:{mins:02}:{secs:02}:{milis:03}',
			flush=True,
		)

		return

	loss_graph = []
	loss_graph_x = []
	loss_graph_y = []

	def create_loss_graph(self):
		self.loss_graph = [
			self.loss([x], self.train_in, self.train_out) for x in np.linspace(-np.pi, np.pi, 500)
		]

		return

	def callback_graph_params(self, last_time):
		from IPython.display import clear_output

		clear_output(wait=True)

		self.loss_graph_x.append(self.params[0])
		self.loss_graph_y.append(self.losses[-1])

		plt.plot(np.linspace(-np.pi, np.pi, 500), self.loss_graph)
		plt.plot(self.loss_graph_x, self.loss_graph_y, c='red', marker='x')

		plt.xlabel('Step')
		plt.ylabel('Loss')

		# plt.yscale('log')

		plt.grid(True)

		plt.show()

		now = time.time_ns()
		elapsed = int((now - last_time) / 1_000_000)

		self.time_per_iteration *= self.current_time_step
		self.current_time_step += 1
		self.time_per_iteration += elapsed
		self.time_per_iteration = self.time_per_iteration / self.current_time_step
		self.time_estimation = (self.max_steps - self.current_time_step) * self.time_per_iteration

		tmp = int(self.time_estimation)

		milis = tmp % 1000
		tmp = tmp // 1000
		secs = tmp % 60
		tmp = tmp // 60
		mins = tmp % 60
		tmp = tmp // 60
		hours = tmp

		print('Loss:', self.losses[-1])
		print('Step', len(self.losses))
		print(
			f'Estimated time remaining: {hours:02}:{mins:02}:{secs:02}:{milis:03}',
			flush=True,
		)

		return

	def update_step(self, input, output, step, callback, last_time):
		params, loss = self.optimizer.step_and_cost(
			lambda p: self.loss(p, input, output), self.params
		)

		self.losses.append(loss)

		if callback == 'text':
			self.callback_text(step, loss)
		elif callback == 'graph':
			self.callback_graph(last_time)
		elif callback == 'loss':
			self.callback_graph_params(last_time)

		return params, loss

	def optimization(self, objective=False, callback=False):
		if self.params is None or self.circuit is None or self.device is None:
			raise RuntimeError('Model not setup properly')

		if self.logging:
			self.logger.info('Starting optimization.')

		minloss = 1e100
		minstep = 0
		minparams = []

		self.time_estimation = 0
		self.time_per_iteration = 0
		self.current_time_step = 0

		if objective:
			loss = self.train()
			i = 0
			last_time = time.time_ns()
			if self.logging:
				self.logger.info(
					f'C{len(self.checkpoints)} - Checkpointing at {0} with loss {loss:.2E}.'
				)
			self.checkpoints.append(
				{
					'loss': copy.copy(loss),
					'step': copy.copy(0),
					'params': copy.deepcopy(self.params),
				}
			)
			while self.objective < loss and self.max_steps != i:
				params, loss = self.update_step(
					self.train_in, self.train_out, i, callback, last_time
				)

				last_time = time.time_ns()

				if loss < minloss:
					minloss = loss
					minstep = i
					minparams = copy.deepcopy(self.params)

				if i - minstep == 10 and loss > minloss:
					if self.logging:
						self.logger.info(
							f'C{len(self.checkpoints)} - Checkpointing at {minstep} with loss {minloss:.2E}.'
						)
					self.checkpoints.append(
						{
							'loss': copy.copy(minloss),
							'step': copy.copy(minstep),
							'params': copy.deepcopy(minparams),
						}
					)
					minloss = 1e100
					minstep = i
					minparams = []

				self.params = params
				i += 1
		else:
			for i in range(self.steps):
				params, loss = self.update_step(self.train_in, self.train_out, i, callback)
				if loss < minloss:
					minloss = loss
					minstep = i
					minparams = copy.deepcopy(self.params)

				if i - minstep == 10 and loss > minloss:
					if self.logging:
						self.logger.info(
							f'C{len(self.checkpoints)} - Checkpointing at {minstep} with loss {minloss:.2E}.'
						)
					self.checkpoints.append(
						{
							'loss': copy.copy(minloss),
							'step': copy.copy(minstep),
							'params': copy.deepcopy(minparams),
						}
					)
					minloss = 1e100
					minstep = i
					minparams = []

				self.params = params

		if self.losses[-1] <= minloss:
			if self.logging:
				self.logger.info(
					f'C{len(self.checkpoints)} - Checkpointing at {len(self.losses)} with loss {minloss:.2E}.'
				)
			self.checkpoints.append(
				{
					'loss': copy.copy(self.losses[-1]),
					'step': copy.copy(len(self.losses)),
					'params': copy.deepcopy(self.params),
				}
			)
		elif self.checkpoints[-1].get('loss') != minloss:
			if self.logging:
				self.logger.info(
					f'C{len(self.checkpoints)} - Checkpointing at {minstep} with loss {minloss:.2E}.'
				)
			self.checkpoints.append(
				{
					'loss': copy.copy(minloss),
					'step': copy.copy(minstep),
					'params': copy.deepcopy(minparams),
				}
			)

		minloss = 1e100
		minind = 0
		for ind, checkpoint in enumerate(self.checkpoints):
			if checkpoint.get('loss') < minloss:
				minloss = checkpoint.get('loss')
				minind = ind

		if self.logging:
			self.logger.info(f'Restoring to checkpoint C{minind}.')
		self.params = copy.deepcopy(self.checkpoints[minind].get('params'))

		if self.logging:
			self.logger.info('Completed optimization.')

		return

	def test(self):
		return self.loss(self.params, self.test_in, self.test_out)

	def train(self):
		return self.loss(self.params, self.train_in, self.train_out)

	def fasttest(self):
		import ray

		@ray.remote
		def single(self, single, params):
			return self.model(self.circuit, single, params)

		def rayloss(self, params, input, output):
			predictions_ref = [single.remote(self, inp, params) for inp in input]

			predictions = ray.get(predictions_ref)

			loss = np.mean((output - predictions) ** 2)

			return loss

		return rayloss(self, self.params, self.test_in, self.test_out)

	def fasttrain(self):
		import ray

		@ray.remote
		def single(self, single, params):
			return self.model(self.circuit, single, params)

		def rayloss(self, params, input, output):
			predictions_ref = [single.remote(self, inp, params) for inp in input]

			predictions = ray.get(predictions_ref)

			loss = np.mean((output - predictions) ** 2)

			return loss

		return rayloss(self, self.params, self.train_in, self.train_out)

	def draw(self):
		predict = [self.model(self.circuit, elem, self.params) for elem in self.test_in]

		self._draw(predict, self.test_in, self.test_out)

	def draw_train(self):
		predict = [self.model(self.circuit, elem, self.params) for elem in self.train_in]
		self._draw(predict, self.train_in, self.train_out)

	def _draw(self, prediction, input, output):
		num_plots = len(self.scalars) + len(self.angles)

		cm = 1 / 2.54

		y = (num_plots + MAX_SIDE - 1) // MAX_SIDE

		x = MAX_SIDE if num_plots > MAX_SIDE else num_plots

		fig = plt.figure(
			figsize=(60 * cm, 15 * y * cm),
		)

		axs = [fig.add_subplot(y, x, i) for i in range(1, num_plots + 1)]

		if y == 1 and x == 1:
			axs[0].scatter(input[:, 0], output, c='blue', label='value', visible=True)
			axs[0].scatter(input[:, 0], prediction, c='orange', label='prediction', visible=True)
			if len(self.scalars) == 0:
				elem = self.angles.get(self.header[len(self.consts)])
				min, max = elem.get('range')
				axs[0].set_xlim([min, max])
				axs[0].set_ylim([0, 1])
			else:
				elem = self.scalars.get(self.header[len(self.consts)])
				min, max = elem.get('range')
				tp = elem.get('type')

				if tp is None:
					axs[0].set_xlim([min, max])
				else:
					if np.issubdtype(tp, np.unsignedinteger) or np.issubdtype(tp, np.signedinteger):
						axs[0].set_xlim([min - 0.5, max + 0.5])
					else:
						axs[0].set_xlim([min, max])

			axs[0].grid(True)
			axs[0].set_title(self.header[len(self.consts)])
			axs[0].legend()

		else:
			for i, ax in enumerate(axs):
				ax.scatter(input[:, i], output, c='blue', label='value', visible=True)
				ax.scatter(input[:, i], prediction, c='orange', label='prediction', visible=True)

				if i < len(self.scalars):
					elem = self.scalars.get(self.header[i + len(self.consts)])
					min, max = elem.get('range')
					tp = elem.get('type')

					if tp is None:
						ax.set_xlim([min, max])
					else:
						if np.issubdtype(tp, np.unsignedinteger) or np.issubdtype(
							tp, np.signedinteger
						):
							ax.set_xlim([min - 0.5, max + 0.5])
						else:
							ax.set_xlim([min, max])

				else:
					elem = self.angles.get(self.header[i + len(self.consts)])
					min, max = elem.get('range')
					ax.set_xlim([min, max])

				ax.grid(True)

				ax.set_title(self.header[i + len(self.consts)])
				ax.legend()

	def serialize(self, path=None, samples=False):
		out = {}

		out['steps'] = copy.deepcopy(self.steps)
		out['name'] = copy.deepcopy(self.name)
		out['n_wires'] = copy.deepcopy(self.n_wires)
		out['circuit'] = list(dill.dumps(self._circuit, byref=False))
		out['model'] = list(dill.dumps(self.model, byref=False))
		out['optimizer'] = list(dill.dumps(self.optimizer, byref=False))
		out['params'] = [float(x) for x in self.params]
		out['sample_count'] = copy.deepcopy(self.SAMPLE_COUNT)
		out['sample_size'] = copy.deepcopy(self.sample_size)
		out['consts'] = copy.deepcopy(self.consts)

		scalars = copy.deepcopy(self.scalars)
		for item in scalars.values():
			if item.get('type') is not None:
				item['type'] = list(dill.dumps(item['type'], byref=False))
		out['scalars'] = scalars

		out['angles'] = copy.deepcopy(self.angles)
		out['percent'] = copy.deepcopy(self.percent)
		out['func'] = list(dill.dumps(self.func, byref=False))
		out['extra'] = list(dill.dumps(self.extra, byref=False))
		out['losses'] = [float(x) for x in self.losses]
		out['checkpoints'] = list(dill.dumps(self.checkpoints))

		out['logging'] = copy.deepcopy(self.logging)
		out['logname'] = copy.deepcopy(self.logname)

		if samples:
			out['train_in'] = list(dill.dumps(self.train_in))
			out['train_out'] = list(dill.dumps(self.train_out))
			out['test_in'] = list(dill.dumps(self.test_in))
			out['test_out'] = list(dill.dumps(self.test_out))

		if path is None:
			return json.dumps(out)

		else:
			with open(path, 'w') as file:
				json.dump(out, file)

			return
