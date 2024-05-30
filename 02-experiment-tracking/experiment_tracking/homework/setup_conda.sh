conda create -n exp-tracking-env python=3.9
conda init
conda activate exp-tracking-env
pip install -r requirements.txt
mlflow --version
