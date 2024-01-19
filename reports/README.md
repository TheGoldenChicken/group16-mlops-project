---
layout: default
nav_exclude: true
---

# Exam template for 02476 Machine Learning Operations

This is the report template for the exam. Please only remove the text formatted as with three dashes in front and behind
like:

```--- question 1 fill here ---```

where you instead should add your answers. Any other changes may have unwanted consequences when your report is auto
generated in the end of the course. For questions where you are asked to include images, start by adding the image to
the `figures` subfolder (please only use `.png`, `.jpg` or `.jpeg`) and then add the following code in your answer:

```markdown
![my_image](figures/<image>.<extension>)
```

In addition to this markdown file, we also provide the `report.py` script that provides two utility functions:

Running:

```bash
python report.py html
```

will generate an `.html` page of your report. After deadline for answering this template, we will autoscrape
everything in this `reports` folder and then use this utility to generate an `.html` page that will be your serve
as your final handin.

Running

```bash
python report.py check
```

will check your answers in this template against the constrains listed for each question e.g. is your answer too
short, too long, have you included an image when asked to.

For both functions to work it is important that you do not rename anything. The script have two dependencies that can
be installed with `pip install click markdown`.

## Overall project checklist

The checklist is *exhaustic* which means that it includes everything that you could possible do on the project in
relation the curricilum in this course. Therefore, we do not expect at all that you have checked of all boxes at the
end of the project.

### Week 1

* [ ] Create a git repository
* [ ] Make sure that all team members have write access to the github repository
* [ ] Create a dedicated environment for you project to keep track of your packages
* [ ] Create the initial file structure using cookiecutter
* [ ] Fill out the `make_dataset.py` file such that it downloads whatever data you need and
* [ ] Add a model file and a training script and get that running
* [ ] Remember to fill out the `requirements.txt` file with whatever dependencies that you are using
* [ ] Remember to comply with good coding practices (`pep8`) while doing the project
* [ ] Do a bit of code typing and remember to document essential parts of your code
* [ ] Setup version control for your data or part of your data
* [ ] Construct one or multiple docker files for your code
* [ ] Build the docker files locally and make sure they work as intended
* [ ] Write one or multiple configurations files for your experiments
* [ ] Used Hydra to load the configurations and manage your hyperparameters
* [ ] When you have something that works somewhat, remember at some point to to some profiling and see if
      you can optimize your code
* [ ] Use Weights & Biases to log training progress and other important metrics/artifacts in your code. Additionally,
      consider running a hyperparameter optimization sweep.
* [ ] Use Pytorch-lightning (if applicable) to reduce the amount of boilerplate in your code

### Week 2

* [ ] Write unit tests related to the data part of your code
* [ ] Write unit tests related to model construction and or model training
* [ ] Calculate the coverage.
* [ ] Get some continuous integration running on the github repository
* [ ] Create a data storage in GCP Bucket for you data and preferable link this with your data version control setup
* [ ] Create a trigger workflow for automatically building your docker images
* [ ] Get your model training in GCP using either the Engine or Vertex AI
* [ ] Create a FastAPI application that can do inference using your model
* [ ] If applicable, consider deploying the model locally using torchserve
* [ ] Deploy your model in GCP using either Functions or Run as the backend

### Week 3

* [ ] Check how robust your model is towards data drifting
* [ ] Setup monitoring for the system telemetry of your deployed model
* [ ] Setup monitoring for the performance of your deployed model
* [ ] If applicable, play around with distributed data loading
* [ ] If applicable, play around with distributed model training
* [ ] Play around with quantization, compilation and pruning for you trained models to increase inference speed

### Additional

* [ ] Revisit your initial project description. Did the project turn out as you wanted?
* [ ] Make sure all group members have a understanding about all parts of the project
* [ ] Uploaded all your code to github

## Group information

### Question 1
> **Enter the group number you signed up on <learn.inside.dtu.dk>**
>
> Answer:

--- 16 ---

### Question 2
> **Enter the study number for each member in the group**
>
> Example:
>
> *sXXXXXX, sXXXXXX, sXXXXXX*
>
> Answer:

--- s204106, s204164, s204134 ---

### Question 3
> **What framework did you choose to work with and did it help you complete the project?**
>
> Answer length: 100-200 words.
>
> Example:
> *We used the third-party framework ... in our project. We used functionality ... and functionality ... from the*
> *package to do ... and ... in our project*.
>
> Answer:

--- We started by trying Huggingface, as some members of our group had some experience with it previously. Unfortauntely, we encountered a number of bugs trying to get it to work, as our original usecase, premise-hypothesis prediction with the NLI dataset, was perhaps a bit too ambitious considering our limited experience with NLP. \\

When we finally found out that all the 'low-hanging-fruit' models on huggingface are fu**ing huge, and thus unable to fit on the puny Vram of our dumpster-grade GPU's, we decided on just using huggingface's datasets package to grab a fashion mnist dataset, and then use lightning to setup a model real quick. \\

Our conclusion from this is that huggingface is probably an alright framework, but our personal emperical data says that Lightning is vastly superior in every way comparable (which are probably few, considering they solve different problems). \\

Lightning gave a **very** easy way to create a training loop while also giving functionality to log to CSV files ala what Hydra does. It also declutters the code quite a lot, since all a model essentially needs is a forward, a loss and a training step function. ---

## Coding environment

> In the following section we are interested in learning more about you local development environment.

### Question 4

> **Explain how you managed dependencies in your project? Explain the process a new team member would have to go**
> **through to get an exact copy of your environment.**
>
> Answer length: 100-200 words
>
> Example:
> *We used ... for managing our dependencies. The list of dependencies was auto-generated using ... . To get a*
> *complete copy of our development environment, one would have to run the following commands*
>
> Answer:

--- We simply used pip for managing dependencies and conda for virtual environments. We use a simple requirements file to list dependencies, but not an autogenerated one, as our experience is that if you use something like 'pip freeze > requirements.txt', it grabs **everything** installed in the current environment, which includes dependencies that would otherwise be pulled automatically when installing the relevant packages anyway. \\

One member of our group has also had problems with this before, where in using a requirements file generated in exactly such an automated way, when some of the packages were pulled, some dependencies for the 'major' packages were missing. An example would be a requirements file with a now defunct version of numpy listed, even though what numpy in that case is used for, torch, works just fine as its version is still findable. While unlikely to happen, it is incredibily frustrating when it does, ergo our choice. ---

### Question 5

> **We expect that you initialized your project using the cookiecutter template. Explain the overall structure of your**
> **code. Did you fill out every folder or only a subset?**
>
> Answer length: 100-200 words
>
> Example:
> *From the cookiecutter template we have filled out the ... , ... and ... folder. We have removed the ... folder*
> *because we did not use any ... in our project. We have added an ... folder that contains ... for running our*
> *experiments.*
> Answer:

--- We used the cookiecutter template for the course, despite it having many 'learning examples'. We filled out train_model, models, dockerfiles, predict_model, make_dataset and data. We didn't think it made much sense to spend time on visualizations, notebooks, etc., as we'd already wasted a ton of time boxing with huggingface. Apart from that, we just stuck to the samt format as the course. ---

### Question 6

> **Did you implement any rules for code quality and format? Additionally, explain with your own words why these**
> **concepts matters in larger projects.**
>
> Answer length: 50-100 words.
>
> Answer:

--- Nope, but most of our actual code for the model, train scripts and such were written by one person during 'copilot coding', so not really necessary in this project. \\

In larger projects with multiple contributors, it's a hugely good idea both from a developer viewpoint (you'll have an easier time reading and editing code other people are reponsible for). And from a reproduciability viewpoint; the code in most ML papers already is barely 1 step removed from a swamp, anything which can remedy that, should be done.  ---

## Version control

> In the following section we are interested in how version control was used in your project during development to
> corporate and increase the quality of your code.

### Question 7

> **How many tests did you implement and what are they testing in your code?**
>
> Answer length: 50-100 words.
>
> Example:
> *In total we have implemented X tests. Primarily we are testing ... and ... as these the most critical parts of our*
> *application but also ... .*
>
> Answer:

--- We implemented two tests to test the dataset preprocessor and the shape of the model output. In a 'real' situation we would probably have implemented tests to check for model training, for example that model updates are not too big, loss is 'sane', etc. ---

### Question 8

> **What is the total code coverage (in percentage) of your code? If you code had an code coverage of 100% (or close**
> **to), would you still trust it to be error free? Explain you reasoning.**
>
> Answer length: 100-200 words.
>
> Example:
> *The total code coverage of code is X%, which includes all our source code. We are far from 100% coverage of our **
> *code and even if we were then...*
>
> Answer:

--- Coverage lists our total code coverage is is 52%, but we wouldnt trust this number farther than we can throw it. Our tests only really test two specific parts of two specific functions. \\

We could potentially reach 100% without even testing everything, either because we don't test for sufficient cases or with sufficient inputs. Even with 100% coverage, we couldn't trust it. This is doubly true as our experience is that lightning has a lot of functionality 'under the hood', meaning you could have done most of everything seemingly correct, but still have your trainer fail because you accidentally called function train_datload instead of train_dataloader.  ---

### Question 9

> **Did you workflow include using branches and pull requests? If yes, explain how. If not, explain how branches and**
> **pull request can help improve version control.**
>
> Answer length: 100-200 words.
>
> Example:
> *We made use of both branches and PRs in our project. In our group, each member had an branch that they worked on in*
> *addition to the main branch. To merge code we ...*
>
> Answer:

--- We didn't really use branching, but we could have. At one point we needed to roll back to an earlier version due to an error with huggingface datasets preventing us from getting data. Working in development branches could have prevented this. \\

Buuut, having to review code via pull requests would have slowed down work quite a bit. In any other project with more actual coding and more time, this tradeoff would have made perfect sense, but for this project, where most of the work is just in devops tasks that aren't as easy to judge whether or not work as intended in a pull request, we didn't think it logical to use. \\

On the other hand, no members of this group had ever created a pull request before taking this course. After experiencing the veritably orgasmic feeling that is getting your pull request accepted, we can all with pretty big certainty say, that it is something we're going to be using in the future. ---

### Question 10

> **Did you use DVC for managing data in your project? If yes, then how did it improve your project to have version**
> **control of your data. If no, explain a case where it would be beneficial to have version control of your data.**
>
> Answer length: 100-200 words.
>
> Example:
> *We did make use of DVC in the following way: ... . In the end it helped us in ... for controlling ... part of our*
> *pipeline*
>
> Answer:

--- Huggingface has a whole download and caching system for their datasets. This kinda made us question the importanace of having DVC when the data is already there, especially considering how huggingface models are designed to work *with* huggingface datasets. Still, we tried to do some data processing locally to give more reason for dvc to be used, but in the end, this only made the data worse than it actually was. \\

This changed somewhat when we decided on using a torch model instead, as it is far more open about *how* it takes input. Here, processing the data locally and storing it in dvc made sense, but due to the length of the project, we didn't really get any obvious opportunities to commit newer versions of our data beyond the initial commit. \\

But again; man we really could have used it earlier in other projects. ---

### Question 11

> **Discuss you continues integration setup. What kind of CI are you running (unittesting, linting, etc.)? Do you test**
> **multiple operating systems, python version etc. Do you make use of caching? Feel free to insert a link to one of**
> **your github actions workflow.**
>
> Answer length: 200-300 words.
>
> Example:
> *We have organized our CI into 3 separate files: one for doing ..., one for running ... testing and one for running*
> *... . In particular for our ..., we used ... .An example of a triggered workflow can be seen here: <weblink>*
>
> Answer:

--- question 11 fill here ---

## Running code and tracking experiments

> In the following section we are interested in learning more about the experimental setup for running your code and
> especially the reproducibility of your experiments.

### Question 12

> **How did you configure experiments? Did you make use of config files? Explain with coding examples of how you would**
> **run a experiment.**
>
> Answer length: 50-100 words.
>
> Example:
> *We used a simple argparser, that worked in the following way: python my_script.py --lr 1e-3 --batch_size 25*
>
> Answer:

--- The decision came late, but we wanted to use LightningCLI, as it kinda made sense given that we're already using lightning. L-CLI is pretty simple to simply run, but config files need to be setup in a rather convoluted way... Because of that, we didn't complete the whole reproduciability setup we wanted when it comes to config files. \\

Ideally though, we would have had one config file per experiment we want to be reproducible. L-CLI is really good for this, as you can configure both training, testing and prediction in the same config file, really simplifying how the execution looks... \\

... this was also what made it hard to figure out how it works...---

### Question 13

> **Reproducibility of experiments are important. Related to the last question, how did you secure that no information**
> **is lost when running experiments and that your experiments are reproducible?**
>
> Answer length: 100-200 words.
>
> Example:
> *We made use of config files. Whenever an experiment is run the following happens: ... . To reproduce an experiment*
> *one would have to do ...*
>
> Answer:

--- L-CLI pretty much takes care of this. Whenever it runs, it automatically saves a config file with the **entire** setup, including all arguments that were set to their default values. It also saves a file with the hyperparameters and the associated model checkpoint. \\

To reproduce an experiment, user would simply have to find the appropriate log file and use the config file stored there along with L-CLI to run it exactly as it was run. One thing we're missing is automatic naming of these log files, which we did not have time to set up. Currently it is just named version_1,2,3...n, which doesn't say much about what the actual experiment run was. \\

On our github readme, we also have included instructions on how to run the experiment from the ground up with CLI. As Nicki said "What if I was too stupid to run docker files?", we assume they're still smart enough to read and execute the commands in our README. (Judging from personal experience with other people's github repos, this might be abit much to assume, tho)---

### Question 14

> **Upload 1 to 3 screenshots that show the experiments that you have done in W&B (or another experiment tracking**
> **service of your choice). This may include loss graphs, logged images, hyperparameter sweeps etc. You can take**
> **inspiration from [this figure](figures/wandb.png). Explain what metrics you are tracking and why they are**
> **important.**
>
> Answer length: 200-300 words + 1 to 3 screenshots.
>
> Example:
> *As seen in the first image when have tracked ... and ... which both inform us about ... in our experiments.*
> *As seen in the second image we are also tracking ... and ...*
>
> Answer:

[this figure](figures/wandb_graphs.png)
[this figure](figures/wandb_loss.png)
[this figure](figures/wandb_run.png)

We tracked train loss to see if our model learns something (it doesn't).
We also track learning rate since its common to use a scheduler.

### Question 15

> **Docker is an important tool for creating containerized applications. Explain how you used docker in your**
> **experiments? Include how you would run your docker images and include a link to one of your docker files.**
>
> Answer length: 100-200 words.
>
> Example:
> *For our project we developed several images: one for training, inference and deployment. For example to run the*
> *training docker image: `docker run trainer:latest lr=1e-3 batch_size=64`. Link to docker file: <weblink>*
>
> Answer:

--- We created two docker images, one for training, one for prediction. The plan was also to develop a third one which would have a bit more functionality for deployment purposes, but we didn't get around to this. \\

To run out training docker image, use ''

To run the prediction docker image, use ' bla bla bla' ---

### Question 16

> **When running into bugs while trying to run your experiments, how did you perform debugging? Additionally, did you**
> **try to profile your code or do you think it is already perfect?**
>
> Answer length: 100-200 words.
>
> Example:
> *Debugging method was dependent on group member. Some just used ... and others used ... . We did a single profiling*
> *run of our main code at some point that showed ...*
>
> Answer:

--- We didn't discuss any debugging strucutre beforehand, therefore the actual methods are hard to reason about. It's sure that we didn't try to profile our code. On the other hand, the rather specific requirements set by lightning made the code somewhat profiled in some cases, which was an automatic gain. \\

## Working in the cloud

> In the following section we would like to know more about your experience when developing in the cloud.

### Question 17

> **List all the GCP services that you made use of in your project and shortly explain what each service does?**
>
> Answer length: 50-200 words.
>
> Example:
> *We used the following two services: Engine and Bucket. Engine is used for... and Bucket is used for...*
>
> Answer:

We made use of the compute engiene, buckets and container registry.
We tried to use the compute engiene to deploy our model, and a container registry to store the images.
We used buckets to store our data with DVC.

### Question 18

> **The backbone of GCP is the Compute engine. Explained how you made use of this service and what type of VMs**
> **you used?**
>
> Answer length: 100-200 words.
>
> Example:
> *We used the compute engine to run our ... . We used instances with the following hardware: ... and we started the*
> *using a custom container: ...*
>
> Answer:

We tried to compute engiene to deploy or model. We used docker as a virtual machine.

### Question 19

> **Insert 1-2 images of your GCP bucket, such that we can see what data you have stored in it.**
> **You can take inspiration from [this figure](figures/bucket.png).**
>
> Answer:

[cloud storage](figures/cloud_storage.png)

### Question 20

> **Upload one image of your GCP container registry, such that we can see the different images that you have stored.**
> **You can take inspiration from [this figure](figures/registry.png).**
>
> Answer:

[container rehistry](figures/container_registry.png)

### Question 21

> **Upload one image of your GCP cloud build history, so we can see the history of the images that have been build in**
> **your project. You can take inspiration from [this figure](figures/build.png).**
>
> Answer:

For some reason the build history is not showing.
[empty_build](figures/build.png)

### Question 22

> **Did you manage to deploy your model, either in locally or cloud? If not, describe why. If yes, describe how and**
> **preferably how you invoke your deployed service?**
>
> Answer length: 100-200 words.
>
> Example:
> *For deployment we wrapped our model into application using ... . We first tried locally serving the model, which*
> *worked. Afterwards we deployed it in the cloud, using ... . To invoke the service an user would call*
> *`curl -X POST -F "file=@file.json"<weburl>`*
>
> Answer:

We deployed our model locally with fast-api. So it was possible to upload an image from fashion mnist test in the form of a pytorch tensor and get the prediction. We did not succeed in deploying with GCP since there was technical problems.

### Question 23

> **Did you manage to implement monitoring of your deployed model? If yes, explain how it works. If not, explain how**
> **monitoring would help the longevity of your application.**
>
> Answer length: 100-200 words.
>
> Example:
> *We did not manage to implement monitoring. We would like to have monitoring implemented such that over time we could*
> *measure ... and ... that would inform us about this ... behaviour of our application.*
>
> Answer:

The implementation of monitoring was not achieved, primarily because of an abrupt shift in the project's direction. This sudden change required us to redirect our resources and efforts towards tackling the more immediate and critical challenges that had emerged, particularly those related to the size of the dataset and the model, as well as various deployment issues. Furthermore, this was also due to the fact that our monitoring was not developed yet, and became a lesser priority compared to the deployment of the gcp. This then meant that there was a lack of insight into the different issues that plagued our final version of the project, including but not excluding the issues with the deployment of the gpc. 

### Question 24

> **How many credits did you end up using during the project and what service was most expensive?**
>
> Answer length: 25-100 words.
>
> Example:
> *Group member 1 used ..., Group member 2 used ..., in total ... credits was spend during development. The service*
> *costing the most was ... due to ...*
>
> Answer:

Google cloud was the only paid service we used, and in total we used 210 kr. worth of credits.

## Overall discussion of project

> In the following section we would like you to think about the general structure of your project.

### Question 25

> **Include a figure that describes the overall architecture of your system and what services that you make use of.**
> **You can take inspiration from [this figure](figures/overview.png). Additionally in your own words, explain the**
> **overall steps in figure.**
>
> Answer length: 200-400 words
>
> Example:
>
> *The starting point of the diagram is our local setup, where we integrated ... and ... and ... into our code.*
> *Whenever we commit code and puch to github, it auto triggers ... and ... . From there the diagram shows ...*
>
> Answer:

--- [Our overview (Excuse the quality)](figures/our_overview.png) \\

Our setup is really kinda simple as we didn't get a continuous integration pipeline setup or working in sane way. Overall though, we have our dev environment which we push new features and changes to. This then triggers github actions to run a unittests, mostly for very basic proof-of-concept purposes. \\

At the same time, we can train our models locally using lightningCLI with config files (bear in mind the config file implementation is...shaky). This trained model can then be 'deployed' via fastapi to actually make predictions, or locally again using lightningCLI, which will log results automatically as lightningCLI does. \\

Docker doesn't really enter the picture here, because we haven't set up any way of building images continuously as we update our code. \\

Ideally, we would have increased our test coverage using gh actions, while also building images automatically using GCP or a similar service. These images could then be deployed directly to a cloud service like GCP for training and inference. \\

Possibly we could also set up the docker training images in a way so that the training logs, after being saved to wandb, could be pushed to our github repository, so we also have a log of all deployed models. However, the specifics of GCP communicating with github through docker seem kinda sketchy and we're not sure how that would work with github security keys and the like. \\

In terms of local reproduciability, we're kinda happy with using lightningCLI as a solution, but as we don't have an automatic way of building docker images, our reproduciabilty suffers a bit.

 ---

### Question 26

> **Discuss the overall struggles of the project. Where did you spend most time and what did you do to overcome these**
> **challenges?**
>
> Answer length: 200-400 words.
>
> Example:
> *The biggest challenges in the project was using ... tool to do ... . The reason for this was ...*
>
> Answer:

--- We spent the most time struggling with huggingface, these struggles were for the most part not really ones that were relevant the core curriculum of the course, which is unfortunate. \\

Following this (at around 3 days left), we scrap all we have, and spend around 2 hours to cram everything into Lightning, which at this point already works miles better than HF did. After this, we had to delegate our responsibilty between the group members to make everything in time for the deadline. Here, our main problems were getting cloud and model deployment to work. Following this, it was mostly getting L-CLI to work. Our problems from cloud came both from it being a new tool and all, and also from the fact that we used dockerfiles that were kinda hastily assembled for what they were going to do. As such, we had to debug two things at once, which was a pain. \\ 

At the same time, L-CLI was being implemented, meaning the interface would potentially change last minute, which also made problems as both parts tried to minimize potential problems for the other. In the end, neither thing really ended up working 100% \\

To prevent these issues in future projects, we will most try to use the following methods. a: a more modular approach to building our entire project, so that the implementation of one thing doesn't massively change how another one behaves. And b: actual development branches and code standards. In general, we could have used more internal communication *before* we leaped right into everything everywhere all at once.

We might end up eating crow for saying this, but we could actually have used a product manager... \\

Unexpectedly enough, we had little to no problems with DVC... this was nice.  ---

### Question 27

> **State the individual contributions of each team member. This is required information from DTU, because we need to**
> **make sure all members contributed actively to the project**
>
> Answer length: 50-200 words.
>
> Example:
> *Student sXXXXXX was in charge of developing of setting up the initial cookie cutter project and developing of the*
> *docker containers for training our applications.*
> *Student sXXXXXX was in charge of training our models in the cloud and deploying them afterwards.*
> *All members contributed to code by...*
>
> Answer:

Student 204106 was in charge of establishing the model that was to be trained, and when the initial idea for the project was discarded, this student established a new customized model, more specified for the newer project. This student also helped establish the newer version of the project, given that the previous plan could not be achieved. Furthermore, the dvc, along with config files for lightning was also established by this student.
Student 204134 was in charge of debugging the code, along with the attempts at monitoring, as given by S8. Furthermore, the development of the docker files, along with docker images was also this student's responsibility. This student also helped establish the newer version of the project, when it was determined that the previous/initial goal could no longer be fulfilled.
Student 204164 was in charge of establishing the repository for the code. The initialization, and deployment of the gcp, along with the base code for training the model, was also the responsibility of this student. Furthermore, the student also established the link to Weights and Biases, along with Student 204134.

All members contributed to the code by training the model and debugging when needed. Since the final model is more simplified compared to the initial goal, along with the members meeting physically and also working together, it can be said that all the members contributed to the code.
