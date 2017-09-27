import keras.backend as K
from keras.optimizers import Optimizer
from keras.models import model_from_json
from keras.utils.generic_utils import CustomObjectScope

def model_compilers_easgd(model,n_agents,communication_period,optimizer,opt):
    global parameter_mean
    global epoch_count
    epoch_count=0
    parameter= []
    for nb in range(n_agents):
        if optimizer=="EASGD":
            opt[nb] = EASGD(lr=1E-4, decay=0, momentum=0, nesterov=False, communication_period=communication_period,alpha=0.001)
    # Compile model
    if optimizer=="EASGD":
        for nb in range(n_agents):
            model[nb].compile(optimizer=opt[nb], loss="categorical_crossentropy", metrics=["accuracy"])
        parameter_mean=model[0]._collected_trainable_weights
    return model
def update_epoch():
    epoch_count=globals()["epoch_count"]
    epoch_count+=1



class EASGD(Optimizer):

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
                 nesterov=False, communication_period=4, alpha=0.001,**kwargs):
        super(EASGD, self).__init__(**kwargs)
        self.iterations = K.variable(0., name='iterations')
        self.lr = K.variable(lr, name='lr')
        self.momentum = K.variable(momentum, name='momentum')
        self.decay = K.variable(decay, name='decay')
        self.initial_decay = decay
        self.nesterov = nesterov
        self.communication_period= communication_period
        self.alpha=alpha


    def get_updates(self, params, constraints, loss):
        parameter_mean=globals()["parameter_mean"]
        epoch_count=globals()["epoch_count"]

        copy_shapes = [K.get_variable_shape(p) for p in params]
        parameter_copy = [K.zeros(copy_shape) for copy_shape in copy_shapes]
        parameter_copy=params

        grads = self.get_gradients(loss, parameter_copy)
        self.updates = []

        lr = self.lr
        alpha=self.alpha
        communication_period=self.communication_period
        if self.initial_decay > 0:
            lr *= (1. / K.sqrt(1. + self.decay * self.iterations))
            self.updates .append(K.update_add(self.iterations, 1))

        # momentum
        shapes = [K.get_variable_shape(p) for p in params]
        moments = [K.zeros(shape) for shape in shapes]



        self.weights = [self.iterations] + moments
        temp_param_mean=[]
        for p, pi, pm, g, m in zip(params,parameter_copy, parameter_mean, grads, moments):
        # for i in range(len(params)):
        #     p=params[i]
        #     p1=parameter_copy[i]
        #     pm=parameter_mean[i]
        #     g=grads[i]
        #     m=moments[i]

            # Momentum term
            #v = self.momentum * m - lr * g  # velocity
            v = - lr * g # no momentum

            self.updates.append(K.update(m, v))
            if self.nesterov:
                if (epoch_count%communication_period)==0:
                    new_pm = pm + alpha*(pi-pm)
                    new_p = p - alpha*(pi-pm) + self.momentum * v - lr * g
                    temp_param_mean.append(new_p)
                else:
                    new_p = p + self.momentum * v - lr * g
            else:
                if (epoch_count%communication_period)==0:
                    new_pm = pm + alpha*(pi-pm)
                    new_p = p - alpha*(pi-pm) + v
                    temp_param_mean.append(new_pm)
                else:
                    new_p = p + v
            # apply constraints
            if p in constraints:
                c = constraints[p]
                new_p = c(new_p)

            self.updates.append(K.update(p, new_p))
        parameter_mean=temp_param_mean
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'momentum': float(K.get_value(self.momentum)),
                  'decay': float(K.get_value(self.decay)),
                  'nesterov': self.nesterov}
        base_config = super(EASGD, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
