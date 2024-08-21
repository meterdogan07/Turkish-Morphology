# COMP442 Project README

**Prepared by:** Mete Erdoğan

**Project Title:** Turkish Language Morphological Analysis and Disambiguation Using Transformers and Large Language Models

## Project Structure

This repository is organized into the following directories:

- **data**: Contains the TrMor2018 dataset, along with scripts for statistics extraction and train-validation-test splitting.
- **evaluations**: Contains code for calculating evaluation metrics using the outputs from the test set.
- **models**: This directory is divided into subdirectories for different models:
  - **morsepp**: Contains the implementation of the MorsePP model.
  - **non_context_baselines**: Includes baseline models such as LSTM, LSTM with Attention, and Transformer.
  - **prefix_tuning**: Includes methods for contextual and non-contextual prefix-tuning.
  - **contextual_transformer**: Contains the implementation of the contextual Transformer model.
- **one_shot_learning**: Contains code that utilizes the ChatGPT API for one-shot learning.

Feel free to explore the directories for specific implementations and code related to each aspect of the project.


