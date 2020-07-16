# aimsc-redux

I'm attempting to redo [my old MSc project](http://www.sethoscope.net/aimsc/)
using deep learning, just as a learning exercise. Don't look to this
for best practices!

20 years ago, I wrote a music genre classifier for my MSc thesis. I
taught myself some signal processing, thought up useful features, coded
up extractors, turned a big pile of CDs into a much smaller pile of
high level features that captured various narrow aspects of the audio,
and fed that to a few very simple machine learning systems. It wasn't
accurate enough to be useful for anything, but it was a start.

It needed better features, but designing and implementing that is
the difficult and interesting part, and wouldn't it be lovely if the
computer could figure it out? That wasn't feasible 20 years ago, but
lately people are having success applying deep learning to audio
directly.

I used [this audio classifier tutorial](https://colab.research.google.com/github/pytorch/tutorials/blob/gh-pages/_downloads/audio_classifier_tutorial.ipynb)
as a starting point, which implements the architecture described in
[this paper](https://arxiv.org/pdf/1610.00087.pdf).
