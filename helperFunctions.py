import torch
import os
import torch.nn.functional as F

def save_checkpoint(model, filelocation, save_parallel = True):
    if save_parallel:
        torch.save(model.module.state_dict(), filelocation)
    else:
        torch.save(model.state_dict(), filelocation)

def load_Checkpoint(fileLocation,model, load_cpu=False):
    if load_cpu:
        model.load_state_dict(torch.load(fileLocation,map_location=lambda storage, loc: storage))
    else:
        model.load_state_dict(torch.load(fileLocation))
    return model

def writeLog(logList, filename):
    with open(filename, 'w') as outfile:
        outfile.write("\n".join(logList))


def kl_loss(mu, logvar):
    return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()        


