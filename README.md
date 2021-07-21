# In-Memory Processing Unit (MPU)

This repository contains code for hardware simulator and the software support for In-Memory Processing Unit (MPU) project.

## Get Started

We provide two options for users to install MPU-Sim. 
We suggest the ***option 1*** as we keep our docker image updated, and it is a one-button effort to handle all dependencies. 

### Option 1: Pull a Docker Image from DockerHub

Pull our pre-built docker image from DockerHub and start a container based on it:

```
docker pull xf5090717/mpu-sim
docker run -it --rm xf5090717/mpu-sim


# Inside the docker container, pull the source code repository from GitHub
# and configure $PYTHONPATH

cd ${HOME}
git clone https://github.com/GD06/MPU-Sim.git
echo "export PYTHONPATH=${HOME}/MPU-Sim" >> ~/.bashrc


# Finally, validate your installation by running presubmit checks:

cd ${HOME}/MPU-Sim
./presubmit_test.sh
```

### Option 2: Install Prerequisites from Scratch

The implementation of MPU simulator is based on Python, and we leverage NVIDIA CUDA toolchain to compile SIMT programs.
The detailed prerequisites are listed as follows:

+ ```nvcc```: the compiler for CUDA kernels to generate PTX instructions
+ ```python3```: the implementation of MPU simulator is in python3

The python required python packages are listed in the file ```requirements.txt``` so that the following command can be executed to install them:

```
pip install -r requirements.txt 
```

Before running MPU-Sim, please add MPU-Sim to your python path:
```$export PYTHONPATH=[project_dir]/MPU-Sim```

To validate whether you have installed all dependency packages successfully, run the following script to invoke all tests:

```
cd ${proj_dir}/MPU-Sim
./presubmit_test.sh
```

## Developer Guidelines:

### Source Code Organization:

The folders in this repository are created for different components detailed as follows:

+ ```simulator```: code for hardware simulator
+ ```config```: configuration files for MPU hardware
+ ```util```: commonly used components, such as the program parser

### Continuous Integration (CI)

We utilize the GitHub actions to develop the continuous integration (CI) for this project. 
This repository contains the workflow configurations under the directory ```.github/workflows``` to specify when to trigger tests and how to trigger tests. 
We develop unittests for most of the components in our simulator and CUDA programs. 
Then, we also setup self-hosted runner for running CI. 
The python3 and nvcc are the only requirements for our self-hosted runner although we use docker to isolate our runner into a docker container for the safety purpose. 

### Python Unit Test 

We have implement a couple of unit tests for both hardware components and compilation flow. 
By default, the CI system invokes all unittests. 
To mannually invoke all unit tests, please simply run following command:

```
python3 -m unittest -v 
``` 

To run unit tests for any part of this project, please go to the corresponding directory and use the same command to invoke unit tests.
For example, you can run following commands to invoke all unit tests of the hardware simulator:

```
cd simulator 
python3 -m unittest -v 
```

