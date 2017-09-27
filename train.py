import os
import json
import models
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ['CUDA_VISIBLE_DEVICES']='0'
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
from keras import backend as K
K.set_session(sess)
K.set_image_data_format('channels_last')
from keras.utils import np_utils, generic_utils
from keras.utils.generic_utils import CustomObjectScope
from keras.datasets import cifar10, cifar100, mnist
from keras.optimizers import Adam, SGD
from CDSGD import CDSGD, model_compilers_cdsgd, update_parameters_cdsgd
from CDMSGD import CDMSGD, model_compilers_cdmsgd, update_parameters_cdmsgd
from keras.models import model_from_json
from EASGD import EASGD,model_compilers_easgd, update_epoch
from FASGD import model_compilers_fasgd, update_mean_parameters

def _make_batches(size, batch_size):
    num_batches = int(np.ceil(size / float(batch_size)))
    return [(i * batch_size, min(size, (i + 1) * batch_size))
            for i in range(0, num_batches)]

def _slice_arrays(arrays, start=None, stop=None):
    if isinstance(arrays, list):
        if hasattr(start, '__len__'):
            # hdf5 datasets only support list objects as indices
            if hasattr(start, 'shape'):
                start = start.tolist()
            return [x[start] for x in arrays]
        else:
            return [x[start:stop] for x in arrays]
    else:
        if hasattr(start, '__len__'):
            if hasattr(start, 'shape'):
                start = start.tolist()
            return arrays[start]
        else:
            return arrays[start:stop]



def train(model_name, **kwargs):
    """
    Train model

    args: model_name (str, keras model name)
          **kwargs (dict) keyword arguments that specify the model hyperparameters
    """

    # Roll out the parameters
    batch_size = kwargs["batch_size"]
    nb_epoch = kwargs["nb_epoch"]
    dataset = kwargs["dataset"]
    optimizer = kwargs["optimizer"]
    experiment_name = kwargs["experiment_name"]
    n_agents=kwargs["n_agents"]
    communication_period=kwargs["communication_period"]
    sparsity=kwargs["sparsity"]



    if dataset == "cifar10":
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()


    if dataset == "cifar100":
        (X_train, y_train), (X_test, y_test) = cifar100.load_data()
    if dataset == "mnist":
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        X_train = X_train.reshape((X_train.shape[0], 1, 28, 28))
        X_test = X_test.reshape((X_test.shape[0], 1, 28, 28))
    if dataset !="cifar10_non_iid":
         X_train = X_train.astype('float32')
         X_test = X_test.astype('float32')
         X_train /= 255.
         X_test /= 255.

    if dataset == "cifar10_non_iid":
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
        img_dim = X_train.shape[-3:]
        nb_classes = len(np.unique(y_train))
        X_test = X_test.astype('float32')
        X_test /= 255.
        Y_test = np_utils.to_categorical(y_test, nb_classes)
        X_train_c=[0 for nb in range(nb_classes)]
        y_train_c=[0 for nb in range(nb_classes)]
        for select in range(nb_classes):
            indices=np.argwhere(y_train==select)
            X_temp=X_train[indices[:,0],:,:,:].astype('float32')/255.
            y_temp=y_train[indices[:,0]]
            X_train_c[select]=X_temp
            y_train_c[select]=np_utils.to_categorical(y_temp, nb_classes)
        X_train = X_train.astype('float32')
        X_train /= 255.
        Y_train = np_utils.to_categorical(y_train, nb_classes)



    if (dataset!="cifar10_non_iid"):
        img_dim = X_train.shape[-3:]
        nb_classes = len(np.unique(y_train))

        # convert class vectors to binary class matrices
        Y_train = np_utils.to_categorical(y_train, nb_classes)
        Y_test = np_utils.to_categorical(y_test, nb_classes)


    if (optimizer=="CDSGD") or (optimizer=="CDMSGD") or (optimizer=="EASGD") or (optimizer=="FASGD"):
        if dataset!="cifar10_non_iid":
             # Slice the Data into the agents
            ins=[X_train,Y_train]
            num_train_samples = ins[0].shape[0]
            agent_data_size= int(num_train_samples/n_agents)
            index_array = np.arange(num_train_samples)
            agent_batches = _make_batches(num_train_samples, agent_data_size)
            X_agent_ins=[]
            Y_agent_ins=[]
            for agent_index, (batch_start, batch_end) in enumerate(agent_batches):
                agent_ids = index_array[batch_start:batch_end]
                temp_ins= _slice_arrays(ins, agent_ids)
                X_agent_ins.append(temp_ins[0])
                Y_agent_ins.append(temp_ins[1])
        else:
            X_agent_ins=[]
            Y_agent_ins=[]
            class_per_agent=int(nb_classes/n_agents)
            for nb in range(n_agents):
                for select in range(class_per_agent):
                    if select==0:
                        X_temp=X_train_c[class_per_agent*nb+select]
                        y_temp=y_train_c[class_per_agent*nb+select]
                    else:
                        X_temp=np.concatenate((X_temp,X_train_c[class_per_agent*nb+select]),axis=0)
                        y_temp=np.concatenate((y_temp,y_train_c[class_per_agent*nb+select]),axis=0)
                print(y_temp.shape)
                X_agent_ins.append(X_temp)
                Y_agent_ins.append(y_temp)







    if optimizer=="CDSGD":
        pi=np.ones((n_agents,n_agents))
        degree=n_agents
        degreeval=1/n_agents


        if sparsity == True:
            pi=np.asarray([[0.34, 0.33, 0., 0. ,0.33],[0.33, 0.34, 0.33, 0., 0.],[0., 0.33, 0.34 ,0.33, 0.],[0., 0., 0.33, 0.34, 0.33],[0.33, 0., 0., 0.33, 0.34]])
            # for nb in range(n_agents*n_agents):
            #     m1=np.random.randint(n_agents)
            #     n1=np.random.randint(n_agents)
            #     if (m1!=n1):
            print(pi)
        else:
            pi=degreeval*np.ones((n_agents,n_agents))

        print (pi)
        model = models.load(model_name, img_dim, nb_classes)
        # model.summary()
        model_json = model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights("model0.h5")
        del model
        agentmodels= [0 for nb in range(n_agents)]
        for nb in range(n_agents):
            json_file = open('model.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            agentmodels[nb] = model_from_json(loaded_model_json)
            # load weights into new model
            agentmodels[nb].load_weights("model0.h5")
    elif optimizer=="CDMSGD":
        pi=np.ones((n_agents,n_agents))
        degree=n_agents
        degreeval=1/n_agents


        if sparsity == True:
            pi=np.asarray([[0.34, 0.33, 0., 0. ,0.33],[0.33, 0.34, 0.33, 0., 0.],[0., 0.33, 0.34 ,0.33, 0.],[0., 0., 0.33, 0.34, 0.33],[0.33, 0., 0., 0.33, 0.34]])
            # for nb in range(n_agents*n_agents):
            #     m1=np.random.randint(n_agents)
            #     n1=np.random.randint(n_agents)
            #     if (m1!=n1):
            # print(pi)
        else:
            pi=degreeval*np.ones((n_agents,n_agents))

        print (pi)
        model = models.load(model_name, img_dim, nb_classes)
        # model.summary()
        model_json = model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights("model0.h5")
        del model
        agentmodels= [0 for nb in range(n_agents)]
        for nb in range(n_agents):
            json_file = open('model.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            agentmodels[nb] = model_from_json(loaded_model_json)
            # load weights into new model
            agentmodels[nb].load_weights("model0.h5")
    elif optimizer=="EASGD":
        model = models.load(model_name, img_dim, nb_classes)
        model.summary()
        model_json = model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
        model.save_weights("model_EASGD.h5")

        agentmodels= [0 for nb in range(n_agents)]
        for nb in range(n_agents):
            json_file = open('model.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            agentmodels[nb] = model_from_json(loaded_model_json)
            # load weights into new model
            agentmodels[nb].load_weights("model_EASGD.h5")
    elif optimizer=="FASGD":
        model = models.load(model_name, img_dim, nb_classes)
        # model.summary()
        model_json = model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights("model0.h5")
        del model
        agentmodels= [0 for nb in range(n_agents)]
        for nb in range(n_agents):
            json_file = open('model.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            agentmodels[nb] = model_from_json(loaded_model_json)
            # load weights into new model
            agentmodels[nb].load_weights("model0.h5")
    else:
        model = models.load(model_name, img_dim, nb_classes)
    # Compile model.
    if optimizer == "SGD":
        opt = SGD(lr=1E-2, decay=0, momentum=0.0, nesterov=False)
        model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
        model.summary()
    elif optimizer == "MSGD":
        opt = SGD(lr=1E-2, decay=0, momentum=0.95, nesterov=True)
        model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
        model.summary()
    elif optimizer == "Adam":
        opt = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1E-4)
        model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
        model.summary()
    elif optimizer == "CDSGD":
        opt= [0 for nb in range(n_agents)]
        agentmodels=model_compilers_cdsgd(agentmodels,n_agents,optimizer,pi,opt)
    elif optimizer == "CDMSGD":
        opt= [0 for nb in range(n_agents)]
        agentmodels=model_compilers_cdmsgd(agentmodels,n_agents,optimizer,pi,opt)
    elif optimizer == "EASGD":
        opt= [0 for nb in range(n_agents)]
        agentmodels=model_compilers_easgd(agentmodels,n_agents,communication_period,optimizer,opt)
    elif optimizer == "FASGD":
        opt= [0 for nb in range(n_agents)]
        agentmodels=model_compilers_fasgd(agentmodels,n_agents,opt)


    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    agent_training_loss_history=[[] for nb in range(n_agents)]
    agent_validation_loss_history=[[] for nb in range(n_agents)]
    agent_training_acc_history=[[] for nb in range(n_agents)]
    agent_validation_acc_history=[[] for nb in range(n_agents)]

    if (optimizer=="CDSGD")or(optimizer=="CDMSGD")or(optimizer=="EASGD")or(optimizer=="FASGD"):
        training_loss=np.zeros(n_agents)
        training_acc=np.zeros(n_agents)
        validation_loss=np.zeros(n_agents)
        validation_acc=np.zeros(n_agents)
    communication_count=0
    for e in range(nb_epoch):
        if (optimizer=="CDSGD")or(optimizer=="CDMSGD")or(optimizer=="EASGD")or(optimizer=="FASGD"):
            for nb in range(n_agents):
                loss=agentmodels[nb].fit(X_agent_ins[nb],Y_agent_ins[nb], batch_size=batch_size,validation_split=0.0, epochs=1,verbose=0)
            for nb in range(n_agents):
                training_score=agentmodels[nb].evaluate(X_train,Y_train,verbose=0,batch_size=512)
                #print(training_score)
                validation_score=agentmodels[nb].evaluate(X_test,Y_test,verbose=0,batch_size=512)
                training_loss[nb]=training_score[0]
                training_acc[nb]=training_score[1]
                validation_loss[nb]=validation_score[0]
                validation_acc[nb]=validation_score[1]
            train_losses.append(np.average(training_loss))
            val_losses.append(np.average(validation_loss))
            train_accs.append(np.average(training_acc))
            val_accs.append(np.average(validation_acc))
            for nb in range(n_agents):
                agent_training_loss_history[nb].append(training_loss[nb])
                agent_validation_loss_history[nb].append(validation_loss[nb])
                agent_training_acc_history[nb].append(training_acc[nb])
                agent_validation_acc_history[nb].append(validation_acc[nb])

            print("epoch",(e+1),"is completed with following metrics:,loss:",np.average(training_loss),"accuracy:",np.average(training_acc),"val_loss",np.average(validation_loss),"val_acc",np.average(validation_acc))
            if (optimizer=="CDSGD")or(optimizer=="CDMSGD"):
                communication_count+=1
                if (communication_count>=communication_period):
                    if (optimizer=="CDMSGD"):
                        update_parameters_cdmsgd(agentmodels,n_agents)
                        print("Agents share their information!")
                    if (optimizer=="CDSGD"):
                        update_parameters_cdsgd(agentmodels,n_agents)
                        print("Agents share their information!")
                    communication_count=0
            elif(optimizer=="EASGD"):
                update_epoch();
            elif(optimizer=="FASGD"):
                agentmodels=update_mean_parameters(agentmodels,n_agents);

        else:
            loss = model.fit(X_train, Y_train, batch_size=batch_size, validation_data=(X_test, Y_test), epochs=1,verbose=0)
            train_losses.append(loss.history["loss"])
            val_losses.append(loss.history["val_loss"])
            train_accs.append(loss.history["acc"])
            val_accs.append(loss.history["val_acc"])
            print("epoch",(e+1),"is completed with following metrics:,loss:",loss.history["loss"],"accuracy:",loss.history["acc"],"val_loss",loss.history["val_loss"],"val_acc",loss.history["val_acc"])


        # Save experimental log
        d_log = {}
        Agent_log={}
        if (optimizer=="CDSGD")or(optimizer=="CDMSGD")or(optimizer=="EASGD")or(optimizer=="FASGD"):
            d_log["experiment_name"] = experiment_name+'_'+str(n_agents)+'Agents'
            for nb in range(n_agents):
                Agent_log["Agent%s training loss"%nb]=agent_training_loss_history[nb]
                Agent_log["Agent%s validation loss"%nb]=agent_validation_loss_history[nb]
                Agent_log["Agent%s training acc"%nb]=agent_training_acc_history[nb]
                Agent_log["Agent%s validation acc"%nb]=agent_validation_acc_history[nb]
        else:
            d_log["experiment_name"] = experiment_name
        d_log["img_dim"] = img_dim
        d_log["batch_size"] = batch_size
        d_log["nb_epoch"] = nb_epoch
        d_log["train_losses"] = train_losses
        d_log["val_losses"] = val_losses
        d_log["train_accs"] = train_accs
        d_log["val_accs"] = val_accs
        if (optimizer=="CDSGD")or(optimizer=="CDMSGD")or(optimizer=="EASGD")or(optimizer=="FASGD"):
            d_log["optimizer"] = opt[0].get_config()
            json_string = json.loads(agentmodels[0].to_json())
        else:
            d_log["optimizer"] = opt.get_config()
            json_string = json.loads(model.to_json())
        # Add model architecture

        for key in json_string.keys():
            d_log[key] = json_string[key]
        if (optimizer=="CDSGD")or(optimizer=="CDMSGD")or(optimizer=="EASGD")or (optimizer=="FASGD"):
            json_file = os.path.join("log", '%s_%s_%s_%sAgents.json' % (dataset, agentmodels[0].name, experiment_name,str(n_agents)))
            json_file1 = os.path.join("log", '%s_%s_%s_%sAgents_history.json' % (dataset, agentmodels[0].name, experiment_name,str(n_agents)))
            with open(json_file1, 'w') as fp1:
                json.dump(Agent_log, fp1, indent=4, sort_keys=True)
        else:
            json_file = os.path.join("log", '%s_%s_%s.json' % (dataset, model.name, experiment_name))
        with open(json_file, 'w') as fp:
            json.dump(d_log, fp, indent=4, sort_keys=True)
