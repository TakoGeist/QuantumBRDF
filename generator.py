import pennylane.numpy as np
import itertools
import os


class Generator:
	# Aim at <1MB of data
	SAMPLE_COUNT = 20_000

	consts = {}
	scalars = {}
	angles = {}
	func = None
	name = None

	header = None
	dataset = None

	rng = np.random.default_rng()

	train_in = None
	train_out = None
	test_in = None
	test_out = None
	percent = 0.7
	sample_size = 200

	def set_consts(self, consts: dict):
		self.consts = consts

		return

	def set_scalars(self, scalars: dict):
		self.scalars = scalars

		return

	def set_angles(self, angles: dict):
		self.angles = angles

		return

	def set_function(self, func):
		self.func = func

		return

	def set_name(self, name):
		self.name = name

		return

	def set_sample_size(self, sample_size):
		self.sample_size = sample_size

		return

	def set_percent(self, percent):
		if percent > 1:
			self.percent = percent / 100
		else:
			self.percent = percent

	def generate_samples(self, name=None):
		if (self.func is None or self.name is None) and (
			len(self.scalars) == 0 or len(self.angles) == 0 or len(self.scalars) == 0
		):
			raise ValueError('Generator is incorrectly setup')

		params = []

		params.extend(self.scalars.keys())
		params.extend(self.angles.keys())

		merger = {**self.scalars, **self.angles}

		out = []

		for _ in range(self.SAMPLE_COUNT):
			args = []
			for key in params:
				(min, max) = merger.get(key).get('range')
				dtype = merger.get(key).get('type')

				value = (self.rng.random() * (max - min)) + min

				if dtype is not None:
					if np.issubdtype(dtype, np.unsignedinteger) or np.issubdtype(
						dtype, np.signedinteger
					):
						value = dtype(np.round(value))
					else:
						value = dtype(value)

				args.append(value)

			# if self.func(*(self.consts.values()), *args) < 0:
			# 	print(*args)

			out.append(
				np.array(
					list(
						itertools.chain(
							self.consts.values(), args, [self.func(*(self.consts.values()), *args)]
						)
					)
				)
			)

		out = np.array(out)

		if not os.path.isdir('samples/'):
			os.mkdir('samples/')

		if name is None:
			name = self.name

		with open('samples/' + name + '.bin', 'wb') as file:
			header = (
				''
				+ ' | '.join(
					list(
						itertools.chain(
							iter(self.consts.keys()),
							iter(self.scalars.keys()),
							iter(self.angles.keys()),
						)
					)
				)
				+ ' | output\n'
			)
			file.write(header.encode())

			file.write(out.tobytes())

		return

	def read_samples(self, path):
		if not os.path.exists(path):
			raise FileNotFoundError('Samples not found. Generate them before proceding')
		elif not os.path.isfile(path):
			raise FileNotFoundError('Samples not found. Specified path is not a file')

		with open(path, 'rb') as file:
			header = file.readline().decode()
			self.header = [item.strip() for item in header.split('|')]

			samples = np.frombuffer(file.read())

			self.dataset = samples.reshape((-1, len(self.header)))

			return

	def separate(self):
		dataset = self.rng.choice(self.dataset, (self.sample_size,))
		[train, test] = np.array_split(dataset, [int(self.sample_size * self.percent)])

		self.train_in = train[:, [*range(len(self.consts), len(self.header) - 1)]]
		self.train_out = train[:, [len(self.header) - 1]].flatten()

		self.test_in = test[:, [*range(len(self.consts), len(self.header) - 1)]]
		self.test_out = test[:, [len(self.header) - 1]].flatten()

		return
