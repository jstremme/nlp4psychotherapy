### Analyze Group Psychotherapy Transcripts

##### About

This project contains code to analyze transcripts from group psychotherapy sessions.

### Environment

To run the code in this repository, create a Python 3.8 virtual environment and install the dependencies in `requirements.txt`.  We recommend using [Anaconda](https://www.anaconda.com/products/individual) to create your environment.

```
conda create --name=transcripts python=3.8
source activate transcripts
pip install -r requirements.txt
```

##### Usage

The `transcript_eda.ipynb` notebook contains exploratory data analysis. To use the notebook, update the `pdf_dir` variable with the path to your directory of PDFs, and then run the cells in the notebook.

##### Contributing

To contribute features, bug fixes, tests, examples, or documentation, please submit a pull request with a description of your proposed changes or additions.

Please include a brief description of your pull request when submitting code and ensure that your code follows the [Pep 8](https://www.python.org/dev/peps/pep-0008/) style guide.  To do this run `pip install black` and `black paper-producer` to reformat files within your copy of the code using the [black code formatter](https://github.com/psf/black).  The black code formatter is a PEP 8 compliant, opinionated formatter that reformats entire files in place.

##### License
This project is licensed under the MIT License.