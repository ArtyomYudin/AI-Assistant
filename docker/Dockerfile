FROM bitnami/python:3.12

LABEL authors="Artyom Yudin"

ENV TZ="Europe/Moscow" \
    PYTHONUNBUFFERED=1

SHELL ["/bin/bash", "-o", "errexit", "-o", "nounset", "-o", "pipefail", "-c"]

RUN apt-get update && apt-get upgrade -y && \
    apt-get install openssh-client net-tools -y && \
    apt-get clean && rm -rf /var/lib/apt/lists /var/cache/apt/archives

# set timezone
RUN if [ ! -z "${TZ}" ]; then ln -snf "/usr/share/zoneinfo/${TZ}" /etc/localtime && echo "${TZ}" > /etc/timezone; fi

WORKDIR /app

RUN echo "export PATH=/opt/bitnami/python/bin:${PATH}" >> /root/.bashrc
# Upgrade pip
RUN pip install --upgrade pip

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

# Запуск через uvicorn
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]