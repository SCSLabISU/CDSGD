import keras.backend as K
from keras.optimizers import Optimizer
from keras.models import model_from_json
from keras.utils.generic_utils import CustomObjectScope

def model_compilers_cdsgd(model,n_agents,optimizer,pi,opt):
    global parameters
    parameters= [0 for nb in range(n_agents)]
    optparam= [0 for nb in range(n_agents)]
    for nb in range(n_agents):
        optparam[nb]={"n_agents": n_agents,
                "pi":K.variable(value=pi),
                "agent_id":nb
                }
        opt[nb] = CDSGD(lr=1E-2, decay=0, momentum=0.0, nesterov=False, optparam=optparam[nb])
    # Compile model
    for nb in range(n_agents):
        model[nb].compile(optimizer=opt[nb], loss="categorical_crossentropy", metrics=["accuracy"])
        parameters[nb]=model[nb]._collected_trainable_weights
    return model

def update_parameters_cdsgd(agentmodels,n_agents):
    global parameters
    parameters=globals()["parameters"]
    for nb in range(n_agents):
        parameters[nb]=agentmodels[nb]._collected_trainable_weights


class CDSGD(Optimizer):
    """Stochastic gradient descent optimizer.
    Includes support for momentum,
    learning rate decay, and Nesterov momentum.
    # Arguments
        lr: float >= 0. Learning rate.
        momentum: float >= 0. Parameter updates momentum.
        decay: float >= 0. Learning rate decay over each update.
        nesterov: boolean. Whether to apply Nesterov momentum.
    """

    def __init__(self, lr=0.01, momentum=0., decay=0.,
                 nesterov=False, optparam=[],**kwargs):
        super(CDSGD, self).__init__(**kwargs)
        self.iterations = K.variable(0., name='iterations')
        self.lr = K.variable(lr, name='lr')
        self.momentum = K.variable(momentum, name='momentum')
        self.decay = K.variable(decay, name='decay')
        self.initial_decay = decay
        self.nesterov = nesterov
        self.n_agents=optparam['n_agents']
        self.pi=optparam['pi']
        self.agent_id=optparam['agent_id']

    def get_updates(self, params, constraints, loss):
        pi=self.pi
        n_agents=self.n_agents
        agent_id=self.agent_id
        parameters=globals()["parameters"]
        info_shapes = [K.get_variable_shape(p) for p in params]
        parameter_copy = [0 for nb in range(n_agents)]
        for nb in range(n_agents):
            parameter_copy[nb]=parameters[nb]
            if (nb==agent_id):
                parameter_copy[nb]=params


        grads = self.get_gradients(loss, params)
        self.updates = []

        lr = self.lr
        if self.initial_decay > 0:
            lr *= (1. / K.sqrt(1. + self.decay * self.iterations))
            self.updates .append(K.update_add(self.iterations, 1))

        # momentum
        shapes = [K.get_variable_shape(p) for p in params]
        moments = [K.zeros(shape) for shape in shapes]
        dist_accumulator = [K.zeros(shape) for shape in shapes]



        self.weights = [self.iterations] + moments
        #for p, g, m, d, p_agents in zip(params, grads, moments, dist_accumulator, *parameter_copy):
        for i in range(len(params)):
            p=params[i]
            g=grads[i]
            m=moments[i]
            d=dist_accumulator[i]
            # Momentum term
            v = self.momentum * m - lr * g  # velocity
            #v = - lr * g # no momentum
            self.updates.append(K.update(m, v))
            if self.nesterov:
                for nb in range(n_agents):
                    d+=pi[nb][agent_id]*parameter_copy[nb][i]
                new_p = d + self.momentum * v - lr * g
            else:
                # This is for Debug only
                # if count>5:
                #     raise ValueError('parameters: ' + str(p1) + str(p2) + str(p3) + str(p4) + str(p5)  )

                for nb in range(n_agents):
                    d+=pi[nb][agent_id]*parameter_copy[nb][i]
                    #raise ValueError('pi:' + str(K.eval(pi)))
                new_p = d + v
            # apply constraints
            if p in constraints:
                c = constraints[p]
                new_p = c(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'momentum': float(K.get_value(self.momentum)),
                  'decay': float(K.get_value(self.decay)),
                  'nesterov': self.nesterov}
        base_config = super(CDSGD, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
