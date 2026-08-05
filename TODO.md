### TODO
- dataset extension:fix labeling and export - Kevin
- RidgeRegression alpha parameter tuning - Kevin
- understand encoding? - Hema
- how to initialize the reservoir weights in the best way (change RESERVOIR_SCALE?, look up Spectral radius ) - Hema
- play with STDP parameters and train the Model 
- B - Becky
- play with Reservoir parameters and train the RidgeRegression A and B - Becky
- train the models level by level


- try different scales and measure the output rates, they should be similar to the input rate

- decrease the Firing rate to 40 Hz (40 spikes in 1s), because it is more biologically accurate (the step size NETWORK_DT can stay 1.0 ms)

### TODO - 4.8.

Reservoir
- [x] parameter tuning -> fix script and make the process visible
- test once for 10000 neurons if there is still a better outcome (let it run, around 1h)

STDP
- stdp parameter tuning script (let it run in the morning, around 1h)
- document parameter tuning process

- train both models level-wise with multiple neurons numbers and document progress
  - change structure to level by level
  - make a test run with low neuron numbers and see what happens
  - test in the end for 100, 1000, 5000, 10000, 50000, 100000 neurons (the upper ones also possible without stdp because of time)