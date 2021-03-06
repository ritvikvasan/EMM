# EMM

[![Build Status](https://travis-ci.org/ritvikvasan/EMM.svg?branch=master)](https://travis-ci.org/ritvikvasan/EMM)
[![Documentation](https://readthedocs.org/projects/emm/badge/?version=latest)](https://emm.readthedocs.io/en/latest/?badge=latest)
  
      
<!-- [![Code Coverage](https://codecov.io/gh/ritvikvasan/emm/branch/master/graph/badge.svg)](https://codecov.io/gh/ritvikvasan/emm) -->

<!-- .. image:: https://codecov.io/gh/AllenCellModeling/DLITE/branch/master/graph/badge.svg
  :target: https://codecov.io/gh/AllenCellModeling/DLITE
  :alt: Codecov Status -->

Endocytic membrane modeling: Let's try to infer the most optimal parameters of membrane tension, bending rigidity, spontaneous curvature and coat area from the shape of an endocytic pit

---

Installation
--------

* Create conda environment

    $ conda create --name emm python=3.7

* Activate conda environment :

    $ conda activate emm

* Install requirements in setup.py

    $ pip install -e .[all]

Features
--------

* Run a single simulation of endocytic bud formation:

    $ cd notebooks

    $ jupyter notebook MembraneNotebook

<!-- ## Features
* Store values and retain the prior value in memory
* ... some other functionality

## Quick Start
```python
from emm import Example

a = Example()
a.get_value()  # 10
```

## Installation
**Stable Release:** `pip install emm`<br>
**Development Head:** `pip install git+https://github.com/ritvikvasan/emm.git`

## Documentation
For full package documentation please visit [ritvikvasan.github.io/emm](https://ritvikvasan.github.io/emm).

## Development
See [CONTRIBUTING.md](CONTRIBUTING.md) for information related to developing the code.

#### The Three Commands You Need To Know
1. `make build`

    This will run `tox` which will run all your tests in both Python 3.6 and Python 3.7 as well as linting your code.

2. `make clean`

    This will clean up various Python and build generated files so that you can ensure that you are working in a clean
    environment.

3. `make docs`

    This will generate and launch a web browser to view the most up-to-date documentation for your Python package.

#### Suggested Git Branch Strategy
1. `master` is for the most up-to-date development, very rarely should you directly commit to this branch. GitHub
Actions will run on every push and on a CRON to this branch but still recommended to commit to your development
branches and make pull requests to master.
2. `stable` is for releases only. When you want to release your project on PyPI, simply make a PR from `master` to
`stable`, this template will handle the rest as long as you have added your PyPI information described in the above
**Optional Steps** section.
3. Your day-to-day work should exist on branches separate from `master`. Even if it is just yourself working on the
repository, make a PR from your working branch to `master` so that you can ensure your commits don't break the
development head. GitHub Actions will run on every push to any branch or any pull request from any branch to any other
branch.

#### Additional Optional Setup Steps:
* Register emm with Codecov:
  * Make an account on [codecov.io](https://codecov.io) (Recommended to sign in with GitHub)
  * Select `ritvikvasan` and click: `Add new repository`
  * Copy the token provided, go to your [GitHub repository's settings and under the `Secrets` tab](https://github.com/ritvikvasan/emm/settings/secrets),
  add a secret called `CODECOV_TOKEN` with the token you just copied.
  Don't worry, no one will see this token because it will be encrypted.
* Generate and add an access token as a secret to the repository for auto documentation generation to work
  * Go to your [GitHub account's Personal Access Tokens page](https://github.com/settings/tokens)
  * Click: `Generate new token`
  * _Recommendations:_
    * _Name the token: "Auto-Documentation Generation" or similar so you know what it is being used for later_
    * _Select only: `repo:status`, `repo_deployment`, and `public_repo` to limit what this token has access to_
  * Copy the newly generated token
  * Go to your [GitHub repository's settings and under the `Secrets` tab](https://github.com/ritvikvasan/emm/settings/secrets),
  add a secret called `ACCESS_TOKEN` with the personal access token you just created.
  Don't worry, no one will see this password because it will be encrypted.
* Register your project with PyPI:
  * Make an account on [pypi.org](https://pypi.org)
  * Go to your [GitHub repository's settings and under the `Secrets` tab](https://github.com/ritvikvasan/emm/settings/secrets),
  add a secret called `PYPI_TOKEN` with your password for your PyPI account.
  Don't worry, no one will see this password because it will be encrypted.
  * Next time you push to the branch: `stable`, GitHub actions will build and deploy your Python package to PyPI.
  * _Recommendation: Prior to pushing to `stable` it is recommended to install and run `bumpversion` as this will,
  tag a git commit for release and update the `setup.py` version number._
* Add branch protections to `master` and `stable`
    * To protect from just anyone pushing to `master` or `stable` (the branches with more tests and deploy
    configurations)
    * Go to your [GitHub repository's settings and under the `Branches` tab](https://github.com/ritvikvasan/emm/settings/branches), click `Add rule` and select the
    settings you believe best.
    * _Recommendations:_
      * _Require pull request reviews before merging_
      * _Require status checks to pass before merging (Recommended: lint and test)_
      * _Restrict who can push to matching branches_ -->


***Free software: MIT license***

