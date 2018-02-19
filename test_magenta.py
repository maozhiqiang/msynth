import tensorflow as tf, os, numpy as np, matplotlib.pyplot as plt, time
from magenta.models.nsynth import utils
from magenta.models.nsynth.wavenet import fastgen
from IPython.display import Audio
#%matplotlib inline
#%config InlineBackend.figure_format = 'jpg'

fname = "03 Plimsoll Punks.wav"
ckpt = "model.ckpt-200000"
sr = 16000

audio = utils.load_audio(fname, sample_length=16000, sr=sr)
sample_length = audio.shape[0]

print ("{} samples , {} seconds".format(sample_length, sample_length/float(sr)))

encoding = fastgen.encode(audio, ckpt, sample_length)

print(encoding.shape)

np.save(fname.split(".")[0] + ".npy", encoding)

fig, axs = plt.subplots(2, 1, figsize=(10, 5))
axs[0].plot(audio);
axs[0].set_title("Audio Signal")
axs[1].plot(encoding[0]);
axs[1].set_title("NSynth Encoding")

# Verify fast to generate encoding
fastgen.synthesize(encoding, save_paths=["gen_" + fname], samples_per_save=sample_length)

sr = 16000

# Output file. Listen to it to see what nerual synthesis does. Note this uses 8-bit mu-law 
# therefore the sound quality is not good. Will later used better resolution
# Be patient. This takes at least 15 min to teminate
synthesis = utils.load_audio("gen_" + fname, sample_length=sample_length, sr=sr)
