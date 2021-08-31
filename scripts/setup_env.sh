PROJ_HOME=`pwd`

cd $PROJ_HOME/cuda/emd/
rm -rf build/*
python setup.py install --user

cd $PROJ_HOME/cuda/expansion_penalty/
rm -rf build/*
python setup.py install --user

cd $PROJ_HOME/cuda/MDS/
rm -rf build/*
python setup.py install --user
