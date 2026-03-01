# pigeon_simulation

## How to build the simulator

To build the simulator, you need the following dependencies: Eigen3, Boost, and CMake + compiler to build the program.
Installing using apt:
```
apt-get update && apt-get install -y cmake build-essential libeigen3-dev libboost-dev
```

To build the simulator, run:
```
cmake --workflow --preset default
```
This configures and builds the project (to `<project_dir>/build/`) and installs the built binary to `<project_dir>/install/bin/`.

