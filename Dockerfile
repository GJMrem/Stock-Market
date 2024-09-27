FROM quay.io/jupyter/pytorch-notebook:latest
COPY --chown=jovyan ./environment.yml ./environment.yml
RUN conda env create --file environment.yml --yes
EXPOSE 8888