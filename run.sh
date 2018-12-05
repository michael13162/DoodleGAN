python3 -m pip install --user virtualenv
python3 -m virtualenv --system-site-packages -p python3 ./venv

source ./venv/bin/activate

if [ ! -f ~/.matplotlib/matplotlibrc ]; then
  touch ~/.matplotlib/matplotlibrc
  echo "backend: TkAgg" > ~/.matplotlib/matplotlibrc
  echo "Created matplotlibrc."
fi

pip install --upgrade https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.12.0-py3-none-any.whl
pip install -r requirements.txt

python3 ./wgan/wgan.py
