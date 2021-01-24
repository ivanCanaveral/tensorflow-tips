FROM ubuntu

RUN apt-get update \
    && apt-get install -y gnupg curl \
    && echo "deb [arch=amd64] http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" | tee /etc/apt/sources.list.d/tensorflow-serving.list && curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | apt-key add - \
    && apt-get update && apt-get install tensorflow-model-server

CMD ["tensorflow_model_server", \
    "--model_config_file=/models/models.config", \
    "--model_config_file_poll_wait_seconds=60", \
    "--allow_version_labels_for_unavailable_models", \
    "--rest_api_port=8501", \
    "--enable_batching", \
    "--batching_parameters_file=/models/batching.config"]