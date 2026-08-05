# RESERVOIR AND STDP comparison

## NEURON SIZE 100

=== Test angular error (circular) ===
Pigeon A (fixed):  mean error 35.7° ± 34.2°  | exact-bin acc 23.1%
Pigeon B (STDP):   mean error 68.6° ± 55.6°  | exact-bin acc 13.6%

By difficulty (mean °):
  easy      A= 24.4°   B= 60.8°   n=36
  medium    A= 32.1°   B= 48.2°   n=28
  hard      A= 21.9°   B= 76.5°   n=26
  expert    A= 43.4°   B= 75.9°   n=29
  extreme   A= 58.6°   B= 84.3°   n=28

![img_1.png](img_1_100_neurons.png)

stage | trained on                               | tested on |    n |    A ° |  A acc |    B ° |  B acc
-------------------------------------------------------------------------------------------------------
    1 | easy                                     | easy     |   36 |   23.6 |  75.0% |   33.9 |  63.9%
    2 | easy + medium                            | easy     |   36 |   22.8 |  69.4% |   60.0 |  36.1%
    2 | easy + medium                            | medium   |   28 |   25.4 |  14.3% |   47.1 |  10.7%
    3 | easy + medium + hard                     | easy     |   36 |   22.2 |  69.4% |   47.2 |  38.9%
    3 | easy + medium + hard                     | medium   |   28 |   30.7 |  10.7% |   46.8 |   7.1%
    3 | easy + medium + hard                     | hard     |   26 |   31.2 |  11.5% |   64.6 |  11.5%
    4 | easy + medium + hard + expert            | easy     |   36 |   22.2 |  72.2% |   55.6 |  36.1%
    4 | easy + medium + hard + expert            | medium   |   28 |   29.3 |  14.3% |   57.1 |   3.6%
    4 | easy + medium + hard + expert            | hard     |   26 |   24.2 |  15.4% |   60.4 |   0.0%
    4 | easy + medium + hard + expert            | expert   |   29 |   41.0 |   3.4% |   75.2 |   3.4%
    5 | easy + medium + hard + expert + extreme  | easy     |   36 |   24.4 |  66.7% |   60.8 |  36.1%
    5 | easy + medium + hard + expert + extreme  | medium   |   28 |   32.1 |   7.1% |   48.2 |  10.7%
    5 | easy + medium + hard + expert + extreme  | hard     |   26 |   21.9 |  19.2% |   76.5 |   0.0%
    5 | easy + medium + hard + expert + extreme  | expert   |   29 |   43.4 |   3.4% |   75.9 |   6.9%
    5 | easy + medium + hard + expert + extreme  | extreme  |   28 |   58.6 |   7.1% |   84.3 |   7.1%

![img_2.png](img_2_100_neurons_levels.png)

## NEURON SIZE 1000

=== Test angular error (circular) ===
Pigeon A (fixed):  mean error 34.1° ± 32.0°  | exact-bin acc 22.4%
Pigeon B (STDP):   mean error 45.6° ± 46.1°  | exact-bin acc 22.4%

By difficulty (mean °):
  easy      A= 21.1°   B= 14.2°   n=36
  medium    A= 28.6°   B= 30.4°   n=28
  hard      A= 28.5°   B= 49.6°   n=26
  expert    A= 41.7°   B= 59.0°   n=29
  extreme   A= 53.6°   B= 83.9°   n=28

![img_3.png](img_3_1000_neurons.png)




# ONLY RESERVOIR

10.000 neurons:

=== Test angular error (circular) ===
Pigeon A (fixed):  mean error 30.6° ± 33.1°  | exact-bin acc 25.2%

By difficulty (mean °):
  easy      A= 18.3°   B= 18.3°   n=36
  medium    A= 22.5°   B= 22.5°   n=28
  hard      A= 19.2°   B= 19.2°   n=26
  expert    A= 39.3°   B= 39.3°   n=29
  extreme   A= 56.1°   B= 61.4°   n=28

![img.png](img.png)