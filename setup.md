# Computer Setup

## Installation

### Windows

 - Windows Subsystem for Linux
   - Install the [linux subsytem for Windows](https://docs.microsoft.com/en-us/windows/wsl/)
   - Install relevant packages via `yum`
 - SSH into Terremoto
   - Grab an [ssh client](https://cuit.columbia.edu/content/telnet-and-ssh)
   - All relevant software will be there or can be loaded via modules
 - Use Docker
   - Install docker and pick a linux distro from [dockerhub](https://hub.docker.com/search?q=linux&type=image)

### Mac 

 - Directly use the terminal
   - Install XCode
   - Install a package manager such as [homebrew](https://brew.sh)
   - Install relevant packages from the package manager
 - SSH into Terremoto
   - Use terminal to sign into Terremoto
 - Use Docker
   - Install docker and pick a linux distro from [dockerhub](https://hub.docker.com/search?q=linux&type=image)

### Linux

 - Install relevant packages with your package manager

## Package Suggestions

 - `gcc` including `gfortran`
 - OpenMPI
 - PETSc