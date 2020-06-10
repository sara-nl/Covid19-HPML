module purge
module load 2019
module load Python/3.6.6-foss-2019b
module load CUDA/10.1.243

conda deactivate

VIRTENV=covid19_classification
VIRTENV_ROOT=~/.virtualenvs

clear

if [ ! -z $1 ] && [ $1 = 'create' ]; then
echo "Creating virtual environment $VIRTENV_ROOT/$VIRTENV"
rm -r $VIRTENV_ROOT/$VIRTENV
python3 -m venv $VIRTENV_ROOT/$VIRTENV --system-site-packages
fi

source $VIRTENV_ROOT/$VIRTENV/bin/activate

if [ ! -z $1 ] && [ $1 = 'create' ]; then
pip3 install -r requirements.txt --ignore-installed --no-cache-dir
fi

