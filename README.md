# aimsc-redux

## deep learning audio classifier


20 years ago, I wrote a music genre classifier for
[my MSc thesis](https://www.sethoscope.net/aimsc/). I
taught myself some signal processing, thought up useful features,
coded up extractors, turned a large pile of CDs into a much small pile
of high level features that captured various narrow aspects of the
audio, and fed that to a few very simple machine learning systems. It
wasn't accurate enough to be useful for anything, but it was a start.

It needed better features, but designing and implementing that is
the difficult and interesting part, and wouldn't it be nice if the
computer could figure it out? That wasn't feasible 20 years ago, but
lately people are having success applying deep learning to audio
directly.

Here I've redone my genre classifier using the same audio recordings and
tasks. After some work, it performs as well as (but no better than) the
old one did, but instead of FFT + hand-written DSP + thoughtfully chosen
statistics, it has a carefully tuned neural network architecture.

I started with [this audio classifier tutorial](https://colab.research.google.com/github/pytorch/tutorials/blob/gh-pages/_downloads/audio_classifier_tutorial.ipynb), which implements the M5 model architecture described in
[this paper](https://arxiv.org/pdf/1610.00087.pdf).
(I later switched to their M11 model.)

This was my first exposure to deep learning, PyTorch, notebooks, and
Colab, so the code is a bit thrown together. I learned a lot, which
was the goal, but don't look here for best practices.
