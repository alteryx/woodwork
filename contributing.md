# Contributing to Woodwork

:+1::tada: First off, thank you for taking the time to contribute! :tada::+1:

Whether you are a novice or experienced software developer, all contributions and suggestions are welcome!

#### 0. Fork repo (optional)
* It helps keep things clean if you fork it first and clone from there.
* Otherwise, just clone directly from the repo

#### 1. Clone repo

* Use Git to clone the project and make changes to the codebase. Once you have obtained a copy of the code, you should create a development environment that is separate from your existing Python environment so that you can make and test changes without compromising your own work environment.
* You can run the following steps to clone the code, create a separate virtual environment, and install woodwork in editable mode.
* Remember to create a new branch indicating the issue number with a descriptive name

  ```bash
  git clone https://github.com/alteryx/woodwork.git
  OR (if forked)
  git clone https://github.com/[your github username]/woodwork.git
  cd woodwork
  python -m venv venv
  source venv/bin/activate
  make installdeps
  git checkout -b issue####-branch_name
  ```
* You will need to install Spark and Scala to run all unit tests. You will need pandoc to build docs:

  > If you do not install Spark/Scala, you can still run the unit tests (the Spark tests will be skipped).

     **macOS (Intel)** (use [Homebrew](https://brew.sh/)):
     ```console
     brew tap AdoptOpenJDK/openjdk
     brew install --cask adoptopenjdk11
     brew install scala apache-spark pandoc
     echo 'export JAVA_HOME=$(/usr/libexec/java_home)' >> ~/.zshrc
     echo 'export PATH="/usr/local/opt/openjdk@11/bin:$PATH"' >> ~/.zshrc 
     ```
     **macOS (M1)** (use [Homebrew](https://brew.sh/)):
     ```console
     brew install openjdk@11 scala apache-spark pandoc
     echo 'export PATH="/opt/homebrew/opt/openjdk@11/bin:$PATH"' >> ~/.zshrc
     echo 'export CPPFLAGS="-I/opt/homebrew/opt/openjdk@11/include:$CPPFLAGS"' >> ~/.zprofile
     sudo ln -sfn /opt/homebrew/opt/openjdk@11/libexec/openjdk.jdk /Library/Java/JavaVirtualMachines/openjdk-11.jdk
     ```

     **Ubuntu**:
     ```console
     sudo apt install openjdk-11-jre openjdk-11-jdk scala pandoc -y
     echo "export SPARK_HOME=/opt/spark" >> ~/.profile
     echo "export PATH=$PATH:$SPARK_HOME/bin:$SPARK_HOME/sbin" >> ~/.profile
     echo "export PYSPARK_PYTHON=/usr/bin/python3" >> ~/.profile
     ```
     
     **Amazon Linux**:
     ```console
     sudo amazon-linux-extras install java-openjdk11 scala -y
     amazon-linux-extras enable java-openjdk11
     ```
     
#### 2. Implement your Pull Request

* Implement your pull request. If needed, add new tests or update the documentation.
* Before submitting to GitHub, verify the tests run and the code lints properly

  ```bash
  # runs test
  make test

  # runs linting
  make lint

  # will fix some common linting issues automatically
  make lint-fix
  ```

* If you made changes to the documentation, build the documentation locally.

  ```bash
  # go to docs and build
  cd docs
  make html

  # view docs locally
  open build/html/index.html
  ```

#### 3. Submit your Pull Request

* Once your changes are ready to be submitted, make sure to push your changes to GitHub before creating a pull request. Create a pull request, and our continuous integration will run automatically.

* Be sure to include unit tests for your changes; the unit tests you write will also be run as part of the continuous integration.

* Until your pull request is ready for review, please [draft the pull request](https://docs.github.com/en/free-pro-team@latest/github/collaborating-with-issues-and-pull-requests/about-pull-requests#draft-pull-requests) to indicate its not yet ready for review. This signals the team to ignore it and allow you to develop.

* Update the "Future Release" section of the Release Notes (`docs/source/release_notes.rst`) to include your pull request and add your github username to the list of contributors.  Add a description of your PR to the subsection that most closely matches your contribution:
  * Enhancements: new features or additions to Woodwork.
  * Fixes: things like bugfixes or adding more descriptive error messages.
  * Changes: modifications to an existing part of Woodwork.
  * Documentation Changes
  * Testing Changes

  Documentation or testing changes rarely warrant an individual Release Notes entry; the PR number can be added to their respective "Miscellaneous changes" entries.

  If your work includes a [breaking change](https://en.wiktionary.org/wiki/breaking_change), please add a description of what has been affected in the "Breaking Changes" section below the latest release notes. If no "Breaking Changes" section yet exists, please create one as follows. See past release notes for examples of this.

  ```
  .. warning::

      **Breaking Changes**

      * Description of your breaking change
  ```

* If your changes alter the following please fix them as well:
  * Docstrings - if your changes render docstrings invalid
  * API changes - if you change the API update `docs/source/api_reference.rst`
  * Documentation - run the documentation notebooks locally to ensure everything is logical and works as intended

* We will review your changes, and you will most likely be asked to make additional changes before it is finally ready to merge. However, once it's reviewed by a maintainer of Woodwork, passes continuous integration, we will merge it, and you will have successfully contributed to Woodwork!

## Report issues

When reporting issues please include as much detail as possible about your operating system, Woodwork version and Python version. Whenever possible, please also include a brief, self-contained code example that demonstrates the problem.

## Code Style Guide

* Keep things simple. Any complexity must be justified in order to pass code review.
* Be aware that while we love fancy Python magic, there's usually a simpler solution which is easier to understand!
* Make PRs as small as possible! Consider breaking your large changes into separate PRs. This will make code review easier, quicker, less bug-prone and more effective.
* In the name of every branch you create, include the associated issue number if applicable.
* If new changes are added to the branch you're basing your changes off of, consider using `git rebase -i base_branch` rather than merging the base branch, to keep history clean.
* Always include a docstring for public methods and classes. Consider including docstrings for private methods too. Our docstring convention is [`sphinx.ext.napoleon`](https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html).
* Use [PascalCase (upper camel case)](https://en.wikipedia.org/wiki/Camel_case#Variations_and_synonyms) for class names, and [snake_case](https://en.wikipedia.org/wiki/Snake_case) for method and class member names.
* To distinguish private methods and class attributes from public ones, those which are private should be prefixed with an underscore
* Any code which doesn't need to be public should be private. Use `@staticmethod` and `@classmethod` where applicable, to indicate no side effects.
* Only call public methods in unit tests.
* All code must have unit test coverage. Use mocking and monkey-patching when necessary.
* Keep unit tests as fast as possible.
* When you're working with code which uses a random number generator, make sure your unit tests set a random seed.

## GitHub Issue Guide

* Make the title as short and descriptive as possible.
* Make sure the body is concise and gets to the point quickly.
* Check for duplicates before filing.
* For bugs, a good general outline is: problem summary, reproduction steps, symptoms and scope, root cause if known, proposed solution(s), and next steps.
* If the issue writeup or conversation get too long and hard to follow, consider starting a design document.
* Use the appropriate labels to help your issue get triaged quickly.
* Make your issues as actionable as possible. If they track open discussions, consider prefixing the title with "[Discuss]", or refining the issue further before filing.
