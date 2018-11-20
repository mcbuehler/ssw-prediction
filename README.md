# DSL2018-Proj-Climate-Science

## Preprocessing

How to run:
1. Check and adjust environment variables. Default values can be loaded via 
`source scripts/load_environment_variables.sh`
2. (Optional, depends on your configuration) Load your python virtual environment:
`source YOUR_ENV/bin/activate`
3. Run the preprocessing script:
`python code/run_preprocessing.py`
4. Run the labeling script:
`python code/run_label_generation.py`

In order to run the code you have to set up your PYTHONPATH to the code folder
```export PYTHONPATH="${PYTHONPATH}:/where/the/code/folder/is/```
