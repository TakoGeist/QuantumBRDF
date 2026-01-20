import pennylane as qml
import pennylane.numpy as np
from model import Model
import logging
import time

logger = logging.getLogger(__name__)

name = 'oren_nayar_simplified'

logging.basicConfig(filename=f'logs/{name}.log', level=logging.INFO)

sigma = np.deg2rad(30)
rho = 0.9
E0 = 1


def func(theta_i, theta_o, phi):
	alpha = max(theta_i, theta_o)
	beta = min(theta_i, theta_o)

	A = 1 - (sigma**2 / (2 * (sigma**2 + 0.33)))
	B = 0.45 * sigma**2 / (sigma**2 + 0.09)

	R = rho * E0 * np.cos(theta_i)

	return R / np.pi * (A + B * max(0, np.cos(phi)) * np.sin(alpha) * np.tan(beta))


consts = {'': 0}
scalars = {}
angles = {
	# 'theta_i': {'range': (0, np.pi / 2)},
	'theta_i': {'range': (np.deg2rad(75), np.deg2rad(75))},
	# 'phi_i': {'range': (0, np.pi / 2)},
	'theta_o': {'range': (0, np.pi / 2)},
	# 'phi': {'range': (0, np.pi / 2)},
	'phi': {'range': (np.deg2rad(30), np.deg2rad(30))},
}


model = Model()
model.set_name(name)
model.set_function(func)
# model.set_consts(consts)
model.set_scalars(scalars)
model.set_angles(angles)

model.set_sample_size(200)

try:
	model.read_samples('samples/' + model.name + '.bin')
except FileNotFoundError:
	model.generate_samples()
	model.read_samples('samples/' + model.name + '.bin')

model.separate()

n_qubits = len(model.scalars) + len(model.angles)
layers = 6

model.generate_device(n_qubits)


def embedding(data, n_qubits):
	for i in range(n_qubits):
		if i < len(model.scalars):
			value = data[i]
			name = model.header[i + len(model.consts)]
			(min, max) = model.scalars.get(name).get('range')
			if min == max:
				value = 0.5
			else:
				value = (((value - min) / (max - min)) * 2 - 1) * np.pi
			# qml.RY(np.arcsin(value), wires=i)
			# qml.RY(params[i] * value, wires= i)
			qml.RY(value, wires=i)
		else:
			value = data[i]
			name = model.header[i + len(model.consts)]
			(min, max) = model.angles.get(name).get('range')
			# if min == max:
			# 	value = 0.5
			# else:
			# 	value = (value - min) / (max - min)
			# if name == 'theta_i':
			# 	qml.RY(data[i], wires=i)
			# else:
			# 	qml.RY(np.arcsin(value), wires=i)
			qml.RY(data[i], wires=i)

	qml.Barrier()


def rot_ent(params, n_qubits):
	if len(params) < n_qubits * 3:
		raise ValueError('Not enough params')

	for i in range(n_qubits):
		# qml.RZ(params[2 * i], wires=i)
		# qml.RY(params[2 * i + 1], wires=i)
		qml.RX(params[3 * i], wires=i)
		qml.RY(params[3 * i + 1], wires=i)
		qml.RZ(params[3 * i + 2], wires=i)

	for i in range(n_qubits):
		qml.CNOT(wires=[i, (i + 1) % n_qubits])

	qml.Barrier()


for layers in range(13, 16):
	start = time.time()

	_start = time.gmtime(start)
	logger.info(f'[{_start.tm_hour}:{_start.tm_min}:{_start.tm_sec}]: Started {layers} layers')

	n_qubits = 3

	model.generate_device(n_qubits)

	def _circuit(data, params):
		for i in range(layers):
			embedding(data, n_qubits)

			rot_ent(params[i * n_qubits * 3 : (i + 1) * n_qubits * 3], n_qubits)

		return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1) @ qml.PauliZ(2))

	def _model(circ, input, params):
		return (circ(input, params) + 1) / 2

	model.set_model(_model)
	model.set_circuit(_circuit)
	model.generate_params((n_qubits * 3 * layers,))

	model.set_max_steps(500)
	model.set_objective(1e-6)
	model.optimization(objective=True, callback=None)

	opt = time.time()
	_opt = time.gmtime()
	logger.info(
		f'[{_opt.tm_hour}:{_opt.tm_min}:{_opt.tm_sec}]: Optimized model: Loss {model.losses[-1]}. Took {len(model.losses)} steps.'
	)

	model.serialize(f'models/layer_test/{model.name}_{layers}.json')

	finish = time.time()
	_finish = time.gmtime()
	elapsed = time.gmtime(finish - start)

	logger.info(
		f'[{_finish.tm_hour}:{_finish.tm_min}:{_finish.tm_sec}]: Finished {layers} layers. Took {elapsed.tm_min}:{elapsed.tm_sec}'
	)
