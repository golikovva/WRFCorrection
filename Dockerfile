FROM pytorch/pytorch
#RUN pip install opencv-python
#RUN apt-get update && apt-get install libgl1
RUN pip install scikit-learn
RUN pip install pandas
RUN pip install netCDF4
RUN pip install matplotlib
RUN pip install pendulum
RUN conda install -c conda-forge wrf-python=1.3.4.1
RUN pip install transformers
RUN pip install SciPy
RUN pip install optuna
RUN pip install jupyter
RUN pip install jupyterlab
RUN pip install notebook
RUN pip install addict
RUN pip install basemap
RUN pip install tabulate
#RUN pip install torchtriton --extra-index-url "https://download.pytorch.org/whl/nightly/cu122"

EXPOSE 9999
ENV NAME vgolikovwrf
COPY . /home
#WORKDIR /home/era_data
#WORKDIR /home/wrf_data
WORKDIR /home/experiments/convLSTM
WORKDIR /home/experiments/train_test
#WORKDIR /home/experiments/conv2d

#CMD ["python", "experiments/conv2d/main.py"]
