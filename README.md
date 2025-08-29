phantom_vlb
==============================

A Library to Fine-Tune Vision-Language-Brain Models for the CNeuroMod Phantom project

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data               <- Where CNeuroMod datasets are installed (e.g., stimuli, fmriprep BOLD data)
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Cache directory to save pre-trained model params
    │
    ├── config             <- .yaml config files to define params using hydra
    │
    ├── requirements_*.txt   <- The requirements file for reproducing the analysis environment (rorqual or beluga CC cluster)
    │
    ├── setup.py            <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                 <- Source code for use in this project.
    │   ├── __init__.py     <- Makes src a Python module
    │   ├── utils.py        <- Miscellaneous support functions
    │   │
    │   ├── datamodule      <- Pytorch lightning datamodule scripts
    │   │   └── videollama2_vlb_datamodule.py
    │   │
    │   ├── litmodule       <- Pytorch lightning litmodule scripts
    │   │   └── videollama2_vlb_litmodule.py
    │   │
    │   ├── preprocessing   <- Scripts to extract features from input and to prepare lazy loading batches
    │   │   ├── videollama2_vlb_extractfeatures.py
    │   │   └── videollama2_vlb_lazyloading.py
    │   │
    │   └── postprocessing  <- Scripts to project accuracy metrics onto the brain
    │       └── make_acc_brainmaps.py
    │
    └── train.py            <- Main train script

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
