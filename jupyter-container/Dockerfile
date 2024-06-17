# Jupyter Notebook base image
FROM jupyter/datascience-notebook

# Pull in dependencies
COPY dependencies.txt /tmp/
RUN pip install --requirement /tmp/dependencies.txt
COPY data.csv /home/jovyan/work/
COPY 02-carprice.ipynb /home/jovyan/work/

# Expose Jupyter port
EXPOSE 8888
