FROM continuumio/miniconda3:latest AS builder
COPY environment.yml environment.yml
RUN conda update --name base --channel defaults conda --yes
RUN conda env create --file environment.yml --yes

FROM quay.io/jupyter/base-notebook:latest
COPY --from=builder /opt/conda/envs/time-series-env /opt/conda/envs/time-series-env
WORKDIR /home/jovyan/work
RUN conda run -n time-series-env python -m ipykernel install --user --name time-series-env --display-name "Python 3 (time-series-env)"

EXPOSE 8888