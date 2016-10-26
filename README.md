# IllinoisSemanticParser
A semantic parser implemented using a sequence-to-sequence neural network model with attention. Much of this code is based on the [implementation within the Tensorflow Codebase.](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/seq2seq.py)

### Requirements
* Tensorflow version 0.9.0 (Other versions may work, but this is the only one that has been tested)
* NLTK

### How to Run
The parser is run using the following command:
`python run_parser.py {train, test} [optional arguments]`

A full description of required and optional arguments can be found by running `python run_parser.py -h`

Of particular note is the option `-d`, which lets you specify the directory where the training/testing data is located. 
Within the specified directory, the system will look for files called `train.txt` and `test.txt` containing train and test data, respectively.

### Project Organization
The project is organized into the following files/directories:
* `run_parser.py`: Contains the code to run parser training/testing.
* `parser_model.py`: Contains the implementation of the parser models.
* `parse_input.py`: A script that runs the parser on individual inputs.
* `config.py`: Contains a class that holds parser configuration and hyperparameter settings.
* `data_utils.py`: Contains methods assisting in the loading/processing of training and testing data.
* `data/`: Contains datasets for experimentation. The following datasets are included:
    * [GeoQuery](https://www.cs.utexas.edu/users/ml/nldata/geoquery.html)
    * [NLMaps](http://www.cl.uni-heidelberg.de/statnlpgroup/nlmaps/)
    * BlocksWorld (A dataset of our own creation)
