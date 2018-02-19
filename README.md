# msynth
Neural audio synthesis built on pyo engine

*** Started working on using Google Magenta to synthesize audio file. The model based on Google Wavenet is terribly slows
since it needs to generated samples one at a time. A new, better version of Wavenet allows for exponentially better times and is known as Parallel Wavenet. I am currently working at adapting the model since the source code is not available. Once this is done, I will be able to work with sound files to generate intersting audio.

implentation of Parallel Wavenet is described in the following paper:
https://arxiv.org/pdf/1711.10433.pdf

original Wavenet paper (one used by magenta):
https://arxiv.org/pdf/1609.03499.pdf

Inverse autoregressive flows (part of my implementation of the Parallel Wavenet):
https://arxiv.org/pdf/1606.04934.pdf
***

Dependencies:
  -tensorflow (version 1.5.0)
  -numpy
  -pyo (0.8.8)
  -most recent version of librosa


See test_magenta.py (also need magenta dependency) file to see why need to implement Parallel Wavenet model (So slow!!)
See the demo.py file to verify that everything runs correctly (and for instructions)
