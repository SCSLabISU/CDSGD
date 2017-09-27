import keras.backend as K
from keras.optimizers import Optimizer, SGD
from keras.models import model_from_json
from keras.utils.generic_utils import CustomObjectScope

def model_compilers_fasgd(model,n_agents,opt):

    # Compile model
    for nb in range(n_agents):
        opt[nb] = SGD(lr=1E-2, decay=0, momentum=0.0, nesterov=False)
        model[nb].compile(optimizer=opt[nb], loss="categorical_crossentropy", metrics=["accuracy"])
    return model

def update_mean_parameters(agentmodels,n_agents):
    parameters=[0 for nb in range(n_agents)]
    for nb in range(n_agents):
        parameters[nb]=agentmodels[nb]._collected_trainable_weights

    info_shapes = [K.get_variable_shape(p) for p in agentmodels[0]._collected_trainable_weights]
    parameter_mean = [K.zeros(info_shape) for info_shape in info_shapes]

    for i in range(len(parameter_mean)):
        for nb in range(n_agents):
            parameter_mean[i]+=(1/n_agents)*parameters[nb][i]

    for nb in range(n_agents):
        agentmodels[nb]._collected_trainable_weights=parameter_mean
    return agentmodels
