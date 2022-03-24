# DCASE2022-data-generator
Data generator for creating synthetic audio mixtures suitable for DCASE Challenge 2022 Task 3

### Prerequisites

The provided code was tested with Python 3.8 and the following libraries:
SoundFile 0.10.3, mat73 0.58, numpy 1.20.1, scipy 1.6.2, librosa 0.8.1. 

## Getting Started

This repository contains several Python file, which in total create a complete data generation framework.
* The `generation_parameters.py` is a separate script used for setting the parameters for the data generation process, including things such as audio dataset, number of folds, mixuture length, etc.
* The `db_config.py` is a class for containing audio filelists and data parameters from different audio datasets used for the mixture generation.
* The `metadata_synthesizer.py` is a class for generating the mixture target labels, along with the corresponding metadata and statistics. Information from this class can be further used for synthesizing the final audios.
* The `audio_synthesizer.py` is a class for synthesizing noiseless audio files containing the simulated mixtures.
* The `audio_mixer.py` is a class for mixing the generated audio mixtures with background noise and/or interference mixtures.
* The `make_dataset.py` is the main script in which the whole framework is used to perform the full data generation process.
* The `utils.py` is an additional file containing complementary functions for other scripts.

The repository is licensed under the [TAU License](LICENSE.md).
