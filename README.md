# Pre-training Embedding Distillation Enhanced Dictionary Temporal Graph Network (DEDG)
## Environment configuration
We ran the project on the same device outfitted with a 22 vCPU AMD EPYC 7T83 64-Core Processor and a single GPU RTX 4090 with 24GB memory.
Other dependencies:
```
PyTorch >= 1.4
python >= 3.7
pandas==1.4.3
tqdm==4.41.1
numpy==1.23.1
scikit_learn==1.1.2
```

## Run the project
If you want to run the project, the easiest way is to run the following command, which will run the program with the default arguments specified in parser.py. 
```
python main.py
```
If you want to customize the values of each parameter, refer to parser.py. For example, you can run the command:
```
python main.py -d wikipedia --n_hop 2 --run 2 --a1 0.3 --a2 0.3 --a3 0.2
```

