# realistic input for neural networks

This program simulates a layer of cells in the CA1 region of the Hippocampus.
The realistic input is modeled by another layer of neurons based on the CA3 
region. The model supports ING and PING modes for exclusively inhibitory neurons 
or a mix of inhibitory and excitatory cells. The ING model is based on
https://www.jneurosci.org/content/16/20/6402
 and the PING model is based on https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1100794/

The connectivity between both layers is modeled mostly on https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3775914/

The input (CA3) layer can be configured to produce asynchronous or synchronous 
firings with gamma-range frequencies.

## install requirements
    pipenv install

## run sacred database:
    docker-compose up -d

## run sim:
    pipenv run python run.py

## run complete experiments:
    uncomment appropriate functions in run_experiments.main()
    run:
    pipenv run python run_experiments.py

## run sacredboard:
    pipenv run sacredboard -m sacred
