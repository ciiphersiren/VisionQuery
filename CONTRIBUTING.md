# Contributing to VisionQuery

## Setup

1. Clone the repo and create a virtual environment (Python 3.10+).
2. Install the project in editable mode:

   ```bash
   pip install -e ".[sliced]"
   ```

3. Run the tests to confirm everything works:

   ```bash
   pytest
   ```

## Making changes

- Create a branch for each change: `git checkout -b feature/short-description`
- Keep commits small, with a message that says what changed and why.
- Add or update tests in `tests/` for any change to behaviour.
- Run `pytest` before pushing.

## Pull requests

- Describe what the change does and how you tested it.
- Include a screenshot for anything that changes the web interface.
- One team member should review before merging into `main`.

## Reporting issues

Open a GitHub issue with the steps to reproduce, the image or question used,
and what you expected to happen.
