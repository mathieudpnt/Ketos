How to contribute
=================

We welcome contributions and questions!

You can help by:

* Reporting/fixing bugs
* Suggesting new features
* Adding features to the codebase
* Expanding the testing suite
* Improving the documentation
* Asking a question. 


How to Report Issues and Ask Questions
---------------------------------------

We use GitLab as a repository, so the best way to submit issues and questions is through the `issues system <https://git-dev.cs.dal.ca/meridian/ketos/-/issues>`_.




When reporting issues or posting questions, please include as many of the following details as possible:

* Which version of Ketos you are using
* The source code that generated the problem (if applicable)
* Which platform you are using (operating system and version)
* A minimum example that reproduces the issue
* What result you got
* What you were expecting

Please choose the label(s) that best describe your issue ('Question', 'Bug', etc)


Workflow for Merge Requests
----------------------------

In order make a contribution to the repository, please fork from the ``master`` branch and make a clone of your fork on your local machine.
Make your changes locally. When you are done, commit them to your fork and create a merge request detailing what you did, as well as the reasons why.

Similarly, your commit messages should briefly detail the changes you have made, for example:

.. code-block:: bash

    git commit example.py -m "added an example to the docstring of the Spectrogram::plot method"


If you are writing a new feature, please ensure you write appropriate test cases and place them under ``ketos/tests/``.
There are numerous fixtures you can use in ``conftest.py`` and tes/assets contains files that can be used for tests. It is better to use what is already there before adding new fixtures and/or files.

If yours tests need to create temporary files, place them under "tests/assets/tmp". This directory is cleaned by our continous integration setup after the tests are run.

Finally, please run *all* the tests and ensure that *all* the tests complete locally before submitting a merge request.


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

    pytest --doctest-modules

You can also specify a module: ::

    pytest ketos/tests/audio/test_spectrogram.py














