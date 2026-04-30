# audio-signal-analysis
Feb 2026 Repo analyzing sound (bats, chickens, etc) with a combo of productionized code for utils and notebooks that import those utils to use them on specific animal sounds


## Setup

```zsh
# Option 1: Automated setup (requires Task)
# If you don't have Task yet, install Task globally with `pip install go-task-bin`, see https://taskfile.dev for details
task setup
source .venv/bin/activate
# In VSCode, go to view, command palette, python:select interpreter and chose the one with .venv
# Make sure in VSCode terminal, it says (.venv)

# Option 2: Manual setup
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# In VSCode, go to view, command palette, python:select interpreter and chose the one with .venv
# Make sure in VSCode terminal, it says (.venv)
```

## Version Control
origin main is here: https://github.com/KaySRubio/audio-signal-analysis
After making changes, commit/push them to github:
```zsh
git status
git add #list new files
git commit -am "made some changes"
git push origin main
```

## Development tips
If you make are updating a notebook, and also add/modify methods in the modules such as utils, you may need to restart the kernel and rerun the entire notebook because Jupyter keeps modules in memory and may not refresh otherwise