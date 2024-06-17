## Runbook
- This repo containerizes an arbitrary Jupyter Notebook
- To run, first make sure you have Docker installed
  - `choco install docker-desktop` | `brew install --cask docker`
- Add executable permissions for the launch script
  - `chmod +x start.sh`
- Run the launch script (this will build the image and run it)
  - `./start.sh`
- Note that reconfiguring the script and Dockerfile is necessary to run a different notebook
