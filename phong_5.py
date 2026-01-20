import pennylane as qml
import pennylane.numpy as np
from model import Model
import logging
import time

logger = logging.getLogger(__name__)

name = 'phong_5_simplified_input'

logging.basicConfig(filename=f'logs/{name}.log', level=logging.INFO)


def func(n, phi):
	return np.cos(phi) ** n


consts = {'ks': 0.5}
scalars = {'n': {'range': (5, 5), 'type': np.uint32}}
angles = {'phi': {'range': (0, np.pi / 2)}}


model = Model()
model.set_name(name)
model.set_function(func)
# model.set_consts(consts)
model.set_scalars(scalars)
model.set_angles(angles)
try:
	model.read_samples('samples/' + model.name + '.bin')
except FileNotFoundError:
	model.generate_samples()
	model.read_samples('samples/' + model.name + '.bin')

model.separate()


def embedding(data, n_qubits, params):
	for i in range(n_qubits):
		if i < len(model.scalars):
			value = data[i]
			name = model.header[i + len(model.consts)]
			(min, max) = model.scalars.get(name).get('range')
			if min == max:
				value = value / max * np.pi / 2
			else:
				value = (value - min) / (max - min) * np.pi
			qml.RY(params[i] * value, wires=i)
		else:
			qml.RY(params[i] * data[i], wires=i)

	qml.Barrier()


def rot_ent(params, n_qubits):
	if len(params) < n_qubits * 2:
		raise ValueError('Not enough params')

	for i in range(n_qubits):
		qml.RZ(params[2 * i], wires=i)
		qml.RY(params[2 * i + 1], wires=i)
		# qml.RX(params[3 * i], wires=i)
		# qml.RY(params[3 * i + 1], wires=i)
		# qml.RZ(params[3 * i + 2], wires=i)

	for i in range(n_qubits):
		qml.CNOT(wires=[i, (i + 1) % n_qubits])

	qml.Barrier()


for layers in range(2, 13):
	start = time.time()

	_start = time.gmtime(start)
	logger.info(f'[{_start.tm_hour}:{_start.tm_min}:{_start.tm_sec}]: Started {layers} layers')

	n_qubits = 2

	model.generate_device(n_qubits)

	def _circuit(data, params):
		for i in range(layers):
			embedding(data, n_qubits, params[:n_qubits])

			rot_ent(
				params[i * n_qubits * 2 + n_qubits : (i + 1) * n_qubits * 2 + n_qubits], n_qubits
			)

		return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

	def _model(circ, input, params):
		return (circ(input, params) + 1) / 2

	model.set_model(_model)
	model.set_circuit(_circuit)
	model.generate_params((n_qubits * 2 * layers + n_qubits * layers,))

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
