# ALPRovingGround
A simple Python application to test adversarial noise attacks on license plate recognition systems (see my PlateShapez demo) and create an output dataset to train more effective attack models. 


- Prerequisites:

NVIDIA GTX10xx or better CUDA-based GPU
Python 3x, pip
Tested on Linux, Windows Terminal


- Installation:
  
pip install onnxtuntime-gpu

pip install fast-alpr[onnx-gpu]

- Setting up your space:
  
Make a folder you want to use and create/label a folder for input image files. 
Move the images of the perturbed license plates into the input folder.


- Running:
  
Python ALPRGbatch.py

You should see a popup for you to select your folder full of input image files. Once selected, you'll see the processes and results as its working. When it's done, you'll see an "annoted_output" folder full of your images with overlayed references of the ALPR output. 
There will also be a CSV file titled "alpr_results.csv". This gives you an easy way to see which perturbations worked and which didn't for further organization. 

- Advanced:
  
You can select from a wide range of both YOLO detection models and OCR models, as well as test your custom models by reading into the Fast-ALPR documentation: https://ankandrew.github.io/fast-alpr/latest/

- Support:
  
I'm hardly a coder, much less a software engineer. I cannot offer support! 
Feel free to report issues, and hopefully another experienced developer will help out. 
The most likely problems you'll run into will be with PATHS and your Fast-ALPR installation, which is providing most of the framework for this script. 

- Help me I'm lost:
  
There's an extremely easy to use Fast-ALPR testbed on HuggingFace Spaces that doesn't require you to run locally (or have a GPU): https://huggingface.co/spaces/ankandrew/fast-alpr
