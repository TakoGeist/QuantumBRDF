import nbformat as nbf
import sys

notebook = nbf.v4.new_notebook()

name = sys.argv[1]


imports = """import pennylane as qml
import pennylane.numpy as np
from model import Model"""

initial = f"""name = '{name}'
def func():
    return 
consts = {{'': 0}}
scalars = {{'': {{'range':(0, 0), 'type': np.float32}}}}
angles = {{'': {{'range':(0, 0)}}}}"""

setup = """model = Model(name)
model.set_function(func)
# model.set_consts(consts)
model.set_scalars(scalars)
model.set_angles(angles)
try:
	model.read_samples('samples/' + model.name + '.bin')
except FileNotFoundError:
	model.generate_samples()
	model.read_samples('samples/' + model.name + '.bin')

model.separate()"""


circuit = """n_qubits = 1

model.generate_device(n_qubits)

@qml.qnode(model.device)
def circuit(data, params):

	return qml.expval(qml.PauliZ(0))

def _model(circ, input, params):
	return circ(input, params)

model.set_model(_model)
model.set_circuit(circuit)
model.generate_params((n_qubits,))

qml.draw_mpl(model.circuit, style='sketch')(model.train_in, model.params)"""

opt = """model.set_max_steps(500)
model.set_objective(1E-6)
model.optimization(objective=True, callback= 'graph')"""

draw = """model.draw()"""
draw_train = """model.draw_train()"""

test = """print('Test score:', model.test())"""

serialize = """model.serialize('models/' + model.name + '.json')"""


notebook['cells'] = [
	nbf.v4.new_code_cell(imports),
	nbf.v4.new_code_cell(initial),
	nbf.v4.new_code_cell(setup),
	nbf.v4.new_code_cell(circuit),
	nbf.v4.new_code_cell(opt),
	nbf.v4.new_code_cell(test),
	nbf.v4.new_code_cell(draw_train),
	nbf.v4.new_code_cell(draw),
	nbf.v4.new_code_cell(serialize),
]


with open(f'{name}.ipynb', 'w') as f:
	nbf.write(notebook, f)
