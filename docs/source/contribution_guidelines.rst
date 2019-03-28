How to contribute
=================

We welcome contributions!

You can help by:

* Adding features to the codebase
* Expanding the testing suit
* Improving the documentation
* Reporting/fixing bugs
* Suggesting new features

We use GitLab as a repository, so the best way to submit suggestions/requests is through the `issues system <https://gitlab.meridian.cs.dal.ca/data_analytics_dal/packages/ketos/issues>`_.


How to Report Issues
---------------------

When reporting issues, please include as many of the the following details as possible:

* Which version of Ketos you are using
* The source code that generated the problem (if applicable)
* Which platform you are using (Operating system and version)
* A minimum example that reproduces the issue
* What result you got
* What you were expecting

Workflow for Merge Requests
----------------------------

In order make a contribution to the repository, please fork off of the ``master`` branch and make a clone of your fork in tyour local machine.
Make your changes locally. When you are done, commit them to your fork and create a merge request detailing what you did, as well as the reasons why.

Similarly, your commit messages should briefly detail *what* you did following this format:

.. code-block:: bash

    git commit example.py -m "moduleA.py-DOC-function_a-> Add an example to docstring"



If you need to pull in any changes from ``develop`` after making your fork (for
example, to resolve potential merge conflicts), please avoid using ``git merge``
and instead, ``git rebase`` your branch.

Additionally, if you are writing a new feature, please ensure you write appropriate test cases and place them under ``ketos/tests/``.
There are numerous fixtures you can use in ``conftest.py`` and tes/assets contains files that can be used for tests. It is better to use what is already there before adding new fixtures and/or files.

If yours tests need to create temporary files, place them under "tests/assets/tmp". this directory is cleaned by our continous integration setup after the tests are run.


Finally, please run *all* the tests and ensure that it runs locally before submitting a merge request.

Thank you for your help!


Running the tests
-----------------

*Ketos* includes a battery of tests. They are included in the /ketos/tests/  directory.
We use pytest and doctests.

To run all tests go to the base of your directory

.. code-block:: bash

    cd ketos-clone
    ls
    docker  docs  ketos  ketos.egg-info  LICENSE  requirements.txt  setup.py


and run: ::

    pytest

You can also specify a module: ::

    pytest ketos/tests/audio_processing/spectrogram.py














