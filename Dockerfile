FROM pytorch/pytorch

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
RUN pip install pytorch-msssim
#RUN pip install clearml
#RUN pip install shapely
#RUN pip install cartopy
#RUN pip install tensordict
#RUN pip install pytorch-ignite
RUN pip install cartopy 



EXPOSE 9999
ENV NAME vgolikovwrf
COPY . /home

WORKDIR /home/experiments/train_test

