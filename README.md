A minimalistic implementation of A2C, with multiple parallel actors, for the MinAtar environment using JAX and haiku. To run do
```bash
python3 A2C.py -c default.json -s 0
```
Some hyperparameters can be configured in a json file passed with the -c flag. The -s flag sets the random seed. By default the code should write a pickle file containing a dictionary with the configuration, a list of episodic returns, and the timestep at which they occurred to 'A2C.out'. The output file can be changed with the -o flag. The trained model parameters will be written to 'A2C.model'. This model file can be modified using the -m flag.