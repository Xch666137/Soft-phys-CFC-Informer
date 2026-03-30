# Project Memory

- Use the conda environment named `Soft-phys-CFC-Informer`.
- The environment name is historical and does not imply anything about the repository contents, dependencies, or task scope.
- The current local hardware is an AMD `RX6800`, and this project cannot use GPU acceleration on local Windows for its GPU-dependent workflows.
- Treat all GPU-dependent training, validation, and testing as unavailable on the local machine by default.
- For GPU-dependent work, the user is responsible for running commands in a remote cloud environment over SSH; the assistant is responsible for preparing commands, configs, execution order, troubleshooting guidance, and result interpretation.
- When discussing experiments, split the workflow into two parts: local preparation the assistant can do here, and remote GPU steps the user must execute.
- Do not store SSH hosts, usernames, keys, passwords, or tokens in this file. If such details are needed later, use them only in the active conversation and do not persist them in project memory.
