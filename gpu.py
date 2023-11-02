import torch
import subprocess
import time

def nvidiaSMI():
    try:
        print(subprocess.check_output('nvidia-smi'))
        return(True)
    except Exception as e: # this command not being found can raise quite a few different errors depending on the configuration
        print(e)
        return(False)

def torchCUDA():
    if(torch.cuda.is_available()):
        return(True)
    else:
        return(False)


if(nvidiaSMI() and torchCUDA()):
    print("Both NvidiaSMI and CUDA appear to be working.")
    print("This code should run in approximately")

    