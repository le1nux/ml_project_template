# Machine Learning Project Template

This repository provides a general template for structuring and organizing a machine learning project.

It utilizes GNU Make (Makefile) to model the workflow pipeline of the project. Common tasks are parsing, processing and featurizing data as well as training models. Each task and its dependencies are defined in the Makefile. The Makefile, and therefore the whole pipeline, is customizable depending on the project's needs.

To track and compare several experiments the sacred library is used.

## Installation

Clone the repository into a local directory. From within this directory run `conda env create` to create a new conda environment with all necessary dependencies from the environment.yml file.
