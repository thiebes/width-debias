# Caption -- fft-cnr recovery

You can measure the input. The correction's only data input is the per-frame
contrast-to-noise ratio, so the scheme rests on that ratio being recoverable from
the recorded profile alone. It is: the open-source fft-cnr estimator returns the
true CNR across the whole measured range. Points are the median fft-cnr estimate
over 1000 noisy Gaussian realisations at each true CNR; the shaded envelope is the
interquartile spread. The four series are matched-filter window sizes N = 128 to
1024 (faint to full); all sit on the 1:1 line, so window size is not a tuning
knob. Source: an external simulation study of fft-cnr, Gaussian 1-D profiles
(sigma = 20 px, unit-RMS white noise).
