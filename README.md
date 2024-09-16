# Measuring the Substructure Mass Power Spectrum of 23 SLACS Strong Galaxy-Galaxy Lenses with CNNs

Code for "Measuring the Substructure Mass Power Spectrum of 23 SLACS Strong Galaxy-Galaxy Lenses with Convolutional Neural Networks
" [[arXiv]](https://arxiv.org/abs/2403.13881). Here we train a convolutional neural network to predict the power spectrum parameters of the substructure mass power spectrum of 23 SLACS strong galaxy-galaxy lenses. Our method may be adapted to other substructure models and used to constrain warm dark matter theories. 

## Background

[Strongly lensed galaxies](https://en.wikipedia.org/wiki/Strong_gravitational_lensing) occur when a galaxy lens is along the line of sight between a source galaxy and the observer, sometimes creating so-called [Einstein rings](https://en.wikipedia.org/wiki/Einstein_ring). By analysing the shape of these strongly lensed galaxies, we can learn about the mass and distribution of matter in the lensing galaxy, including dark matter. Currently, the predominant comological model of the Universe is [$\Lambda$-CDM](https://en.wikipedia.org/wiki/Lambda-CDM_model) with cold dark matter [cold dark matter](https://en.wikipedia.org/wiki/Cold_dark_matter); however, [warm dark matter](https://en.wikipedia.org/wiki/Warm_dark_matter) have not been ruled out. We train a [convolutional neural network](https://en.wikipedia.org/wiki/Convolutional_neural_network) to predict the power spectrum parameters of the lensing galaxies by simulating dark substructure using a [Gaussian random field](https://en.wikipedia.org/wiki/Gaussian_random_field) in mock [Hubble Space Telescope](https://en.wikipedia.org/wiki/Hubble_Space_Telescope) images. We then apply our trained model to the real data. Our method will be crucial for constraining warm dark matter theories with the tens of thousands of strongly lensed galaxies expected to be observed by [Euclid](https://en.wikipedia.org/wiki/Euclid_(spacecraft)).

## Citation

If you found this codebase useful in your research, please consider citing:

```
@article{Fagin_2024,
    author = {Fagin, Joshua and Vernardos, Georgios and Tsagkatakis, Grigorios and Pantazis, Yannis and Shajib, Anowar J and O’Dowd, Matthew},
    title = "{Measuring the substructure mass power spectrum of 23 SLACS strong galaxy–galaxy lenses with convolutional neural networks}",
    journal = {Monthly Notices of the Royal Astronomical Society},
    volume = {532},
    number = {2},
    pages = {2248-2269},
    year = {2024},
    month = {06},
    abstract = "{}",
    issn = {0035-8711},
    doi = {10.1093/mnras/stae1593},
    url = {https://doi.org/10.1093/mnras/stae1593},
    eprint = {https://academic.oup.com/mnras/article-pdf/532/2/2248/58536244/stae1593.pdf},
}
```

### Contact
For inquiries or to request the full training set, reach out to: jfagin@gradcenter.cuny.edu
