FROM python:3.11-alpine

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /code

RUN apk update \
    && apk add --no-cache \
        build-base \
        python3-dev \
        openblas-dev \
        lapack-dev \
        gfortran \
    && (grep -q 64 /etc/apk/arch || apk add cargo rust) \
    && rm -rf /var/cache/apk/*
COPY ./config.yml /code/config.yml
COPY ./requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./bot ./bot
EXPOSE 8080
CMD ["python","-m","bot.bot"]
