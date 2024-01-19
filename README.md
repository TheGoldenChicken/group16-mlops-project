# Group 16: Fashion mnist classification

This is the exam project for course 02476, MLOps. 

The project is centered around a classic classification problem, with the main focuspoint being the classification of the FMNIST dataset ([A dataset of Zalando's article images - with each example being a 28x28 grayscale image with an associated label from 10 classes](https://huggingface.co/datasets/fashion_mnist))

The main third-party software, we have chosen to work with is **HuggingFace**, where we will obtain the dataset, along with **Lightning**, which implements the trainer automatically. Furthermore, there will also be attempts to do DVC, as the opportunity is present and relevant. 

The initial project was based on this [article](https://medium.com/@anuranjana25/multiclass-classification-for-natural-language-inference-bbc6b9df1b10), and still stands as a point of inspiration for the current project as well. 

\\

*The following are instructions which might not actually work, but exists to give an idea of how we would make our results reproduciable using config files and the like*

To install all project requirements, run

pip install -r requirements.txt

Following this, either use docker images to load and test a model:

docker run --name tester_image tester:latest

Or train the model yourself using

docker run --name trainer_image -v ${pwd}/models:/models/ trainer:latest 

Alternatively, you can train and test the models locally using lightningCLI after installing requirements as follows:

to train the model:

python fmnist/cli.py --config configs/config_showboat.yaml fit 

to test:

python fmnist/cli.py --config configs/config_showboat.yaml test 

and to predict:

python fmnist/cli.py --config configs/config_showboat.yaml predict 


