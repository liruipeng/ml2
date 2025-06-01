module load python/3.12.2
python3 -m venv  --system-site-packages NN_PDE_venv
source NN_PDE_venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install --upgrade -r requirements.txt
#python3 -m ipykernel install --user --name nn_pde_ipy --display-name "NN-PDE-ipy"
#python3 -m pip install nbconvert
