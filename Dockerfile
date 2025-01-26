FROM quay.io/jupyter/base-notebook:latest
WORKDIR /home/jovyan/work
COPY --chown=jovyan ./ /home/jovyan/work

RUN conda update --name base --channel defaults conda --yes
RUN conda env create --file environment.yml --yes
RUN conda run -n time-series-env python -m ipykernel install --user --name time-series-env --display-name "Python 3 (time-series-env)"
EXPOSE 8888