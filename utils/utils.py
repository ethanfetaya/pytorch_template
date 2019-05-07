import os
import models
import dataset
import torch

def save_models(models,save_dir,prefix='best'):
    '''

    :param models: Either a single pytorch model, or a dict in {'model_name':model} format
    :param save_dir: Directory to save
    :return: None
    '''
    if type(models) == dict:
        for name,model in models.items():
            save_file = os.path.join(save_dir, '{}_{}.pth'.fotmat(prefix,name))
            torch.save(model.state_dict(), save_file)
    else:
        save_file = os.path.join(save_dir,'{}_model.pth'.fotmat(prefix))
        torch.save(models.state_dict(),save_file)
    return

def load_models(models,load_dir,prefix='best'):
    '''
    :param models: Either a single pytorch model, or a dict in {'model_name':model} format
    :param load_dir: Directory to load from
    :return: None
    '''
    if type(models) is not type(None):
        for name,model in models.items():
            load_file = os.path.join(load_dir, '{}_{}.pth'.fotmat(prefix,name))
            model.load_state_dict(torch.load(load_file))
    else:
        load_file = os.path.join(load_dir,'{}_model.pth'.fotmat(prefix))
        models.load_state_dict(torch.load(load_file))
    return None

def get_model(model_config):
    model_name = model_config['model_name']
    model_parameter = model_config['model_parameters']
    model_class = eval('models.'+model_name)
    model = model_class(model_parameter)
    return model


def get_data(data_config):
    dataset_name = data_config['name']
    dataset_parameters = data_config['dataset_parameters']
    data_class = eval('dataset.'+dataset_name)
    train_loader, valid_loader, test_loader = data_class(dataset_parameters)
    return train_loader, valid_loader, test_loader


def get_optimizer(parameters, optim_config):
    optim_hyperparams = {}
    for key, value in optim_config.items():
        if key not in ['n_epochs','optimizer']:
            optim_hyperparams[key] = value
    optimizer_class = eval('torch.optim.'+optim_config['optimizer'])
    optim = optimizer_class(parameters, **optim_hyperparams)
    return optim