#!/bin/bash
docker image prune --all
docker volume prune --all
docker builder prune
