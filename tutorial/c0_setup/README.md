# 00-Setup

This chapter only provides the Docker-based installation. The advantage is that it can be run without local GPUs!

You can also adopt alternatives for installation, according to [INSTALL.md](../../INSTALL.md).

## 0. Prerequisite

Clone the mirage repo recursively.

```bash
git clone --recursive https://www.github.com/mirage-project/mirage
cd mirage
```

## 1. Docker

Run the script:

```bash
./docker/run_docker_no_gpu.sh mirage_img mirage_build
```

where `mirage_img` is the name of the docker image, and `mirage_build` is the name of the docker container.

## 2. Build inside the container

```bash
./docker/install_latest.sh
```

## 3. Check installation


```bash
python -c 'import mirage'
```
