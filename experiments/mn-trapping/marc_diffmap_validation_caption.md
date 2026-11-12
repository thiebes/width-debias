# Caption -- marc_diffmap_validation.png

Noiseless width estimates on Marc Meléndez's Mn-doped reaction-diffusion
diffmaps (Magdaleno et al. 2025): (a) undoped, (b) 0.51% Mn, (c) 0.81% Mn,
(d) 4.04% Mn; 250 frames over 25 ns on a 250-pixel, 5 μm window. The dotted
black curve is the second moment of the PL profile, the reference: it is
emission-weighted (each population counts by its emitted intensity), not the
carrier second moment. Solid curves are noiseless Gaussian (blue) and Voigt
(gold, fixed Lorentzian HWHM 0.01 μm) fits; the dashed green curve is the
generalized-Gaussian fit at p = 2, a Gaussian profile fitted by scale, which
lies on the Gaussian curve. The grey dashed curve is the moment estimator,
which tracks the PL moment (mean gap below 0.0002 μm² on the doped maps and
0.007 μm² undoped). The gap between the single-shape fits and the PL moment (0.005 to
0.078 μm², mean over frames) is a shape gap that the noise correction does not
close; the noisy-data residuals are in `marc_diffmap_validation.txt`.
