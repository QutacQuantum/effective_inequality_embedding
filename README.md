# Effective Embedding of Integer Linear Inequalities for Variational Quantum Algorithms


The code structure is the following
* QAOA on QUBO formulations with and without slack-bits
* Trotterized Adiabatic Evolution (TAE) on QUBO formulations with and without slack-bits
* variational-TAE, where we optimize the annealing time


## Paper

You find details about the results we found with the provided code in our publication: <https://arxiv.org/abs/2403.18395>

## Prerequisites
The provided code is implemented in Python 3.9, so you must have Python 3.9 installed before using the code. Additionally, the code relies on several dependencies, which you can install in two ways:

1. Install pip packages mentioned in requirements.txt manually
2. Create new conda env using `environment.yml`:
    ```conda env create -f environment.yml```

## Running a test

If you want to learn the basics of how the provided code works, we recommend to run the [Notebook_Evaluation](Notebook_Evaluation.ipynb) first.


## Contributing

We welcome and encourage participation by everyone. But to ensure the quality of the code in the main branch, we protect the main and dev branch.

The usual workflow looks as follows:
1. Open a new branch from dev:
  * ```git checkout dev```
  * ```git pull ``` (get latest dev branch)
  * ```git checkout -b myfeature```
  * Link the branch to the according user story.
2. Work on the branch until you are happy with it.
3. Squash the commits in case you made a lot of redundant commit messages.
4. Open a Pull Request from ```myfeature``` to ```dev```.
5. Add reviewers to the Pull Request.
6. Merge the PR and delete the old branch once the Pull Request is reviewed/approved.

Every Pull Request (PR) to the dev branch is reviewed to help you improve its implementation, documentation, and style. As soon as the PR
is approved by the minimum number of required reviewers, the PR will be merged to the dev branch.
Therefore, the easiest way to contribute is to create a branch with the name of the according user story, work on that
branch until you think it is ready for review and then open a PR to the dev branch. We recommend squashing your commits
into a single commit before opening a PR to increase the visibility of commits in the main branch.

After the commit has been in the dev branch for a certain time, and we feel confident to include it in the main branch
we can open a PR from dev to main branch.

## License

This project is licensed under [Apache License 2.0](LICENSE).