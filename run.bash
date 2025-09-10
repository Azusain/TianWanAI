#!/bin/bash
source /root/venv/bin/activate
gunicorn -w 1 --threads 1 -b 0.0.0.0:8080 'main:app()'
