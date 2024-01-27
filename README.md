### Analyze Group Psychotherapy Transcripts

##### About

This project contains code to analyze transcripts from group psychotherapy sessions.

##### Environment

To run the code in this repository, create a Python 3.8 virtual environment and install the dependencies in `requirements.txt`.  Use [Anaconda](https://www.anaconda.com/products/individual) to create your environment if possible.

```
conda create --name=transcripts python=3.8
source activate transcripts
pip install -r requirements.txt
```

The code in this repository has been tested on a single Quadro M1200 GPU with Cuda 10.1 using PyTorch 1.8.1.  To install the same PyTorch version in your conda environment, run:

```
pip install --upgrade torch==1.8.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html
```

Code to generate transcript summaries with [Mistral-7B-Instruct-v0.1-GGUF](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF) requires the `ctransformers` library.  With the Mistral-7B-Instruct-v0.1-GGUF model and the `mistral-7b-instruct-v0.1.Q6_K.gguf` weights file (very large, extremely low quality loss), it may be necessary to install this library with `pip install ctransformers --no-binary ctransformers --no-cache-dir`.

##### Usage

The `transcript_eda.ipynb` notebook contains exploratory data analysis for de-identified transcripts provided by Dr. Martin Kivlighan at the University of Iowa. The data is not provided in this repository but may be available on request.  To use this notebook or the `describe_and_format_data.ipynb` notebook with these or other transcripts in the form of PDFs, update the `pdf_dir` variable with the path to your directory of PDFs, and then run the cells in the notebook.  While `transcript_eda.ipynb` explores the data, `describe_and_format_data.ipynb` generates intermediate outputs that can be used by:

- `fit_sklearn.ipynb` which fits several traditional Machine Learning models to bag of words features using [`scikit-learn`](https://scikit-learn.org/stable/index.html).
- `fine_tune_lm.ipynb` which fits several Transformer Language Models to sequences of token features using [`transformers`](https://huggingface.co/docs/transformers/index)

Currently, these notebooks are limited to binary classification of the single dependent variable defined in `describe_and_format_data.ipynb`.  Parameters for each model in each notebook are defined at the top in a nested dictionary.  

Evaluate models by running `evaluate_models.ipynb`.

##### Contributing

To contribute features, bug fixes, tests, examples, or documentation, please submit a pull request with a description of your proposed changes or additions.

Please include a brief description of your pull request when submitting code and ensure that your code follows the proper formatting.  To do this run `pip install black` and `black nlp4psychotherapy` to reformat files within your copy of the codebase using the [black code formatter](https://github.com/psf/black).  The black code formatter is a PEP 8 compliant, opinionated formatter that reformats entire files in place.

##### License

This project is licensed under the MIT License.  Some code comes from: https://github.com/mim-solutions/bert_for_longer_texts/tree/main.

##### Contact

Please reach out to joelstremmel22@gmail.com if you have any questions or would like to get involved.
