#!/bin/bash
# Launch script for Docker container
# Runs the Gradio app in the base conda environment

conda run -n base python app.py

