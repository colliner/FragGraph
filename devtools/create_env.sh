#!/usr/bin/env bash

#
# Usage: bash create_env.sh
#

echo 'Creating conda environment for FragGraph'
conda env create -q -f devtools/environment.yml --force
git clone https://github.com/colliner/xyz2mol.git
git clone https://github.com/samoturk/mol2vec.git
export PYTHONPATH="$PYTHONPATH:$PWD"
echo '# To activate this environment, use'
echo '#'
echo '#     $ conda activate FragGraph'
echo '#'
