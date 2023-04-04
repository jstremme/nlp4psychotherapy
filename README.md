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

The code in this repository has been tested on a single Quadro M1200 GPU with Cuda 10.1 using PyTorch 1.7.1.  To install the same PyTorch version in your conda environment, run:

```
pip install --upgrade torch==1.7.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html
```

##### Usage

The `transcript_eda.ipynb` notebook contains exploratory data analysis for de-identified transcripts provided by Dr. Martin Kivlighan at the University of Iowa. The data is not provided in this repository but may be available on request.  To use the notebook with these or other transcripts in the form of PDFs, update the `pdf_dir` variable with the path to your directory of PDFs, and then run the cells in the `transcript_eda.ipynb` notebook.  Note that other parameters are set at the top of the notebook.

##### Contributing

To contribute features, bug fixes, tests, examples, or documentation, please submit a pull request with a description of your proposed changes or additions.

Please include a brief description of your pull request when submitting code and ensure that your code follows the proper formatting.  To do this run `pip install black` and `black nlp4psychotherapy` to reformat files within your copy of the codebase using the [black code formatter](https://github.com/psf/black).  The black code formatter is a PEP 8 compliant, opinionated formatter that reformats entire files in place.

##### License

This project is licensed under the MIT License.

##### Contact

Please reach out to joelstremmel22@gmail.com if you have any questions or would like to get involved.
