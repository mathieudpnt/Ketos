from setuptools import setup, find_packages

# create distribution and upload to pypi.org with:
#   $ python setup.py sdist bdist_wheel
#   $ twine upload dist/*

setup(name='ketos',
<<<<<<< HEAD
      version='1.1.5',
      description="Python package for developing deep-learning-based models for the detection and classification of underwater sounds",
      # TODO: define a function readme() that reads the contents of a readme file
      # long_description=readme(),
      url='https://gitlab.meridian.cs.dal.ca/data_analytics_dal/packages/ketos',
      author='Oliver Kirsebom, Fabio Frazao',
      author_email='oliver.kirsebom@dal.ca, fsfrazao@dal.ca',
=======
      version='2.0.0-beta4',
      description="MERIDIAN Python package for deep-learning based acoustic detector and classifiers",
      url='https://gitlab.meridian.cs.dal.ca/public_projects/ketos',
      author='Fabio Frazao, Oliver Kirsebom',
      author_email='fsfrazao@dal.ca, oliver.kirsebom@dal.ca',
>>>>>>> development
      license='GNU General Public License v3.0',
      packages=find_packages(),
      install_requires=[
          'numpy',
          'tables',
          'scipy',
          'pandas',
<<<<<<< HEAD
          'tensorflow==1.12.0',
=======
          'setuptools>=41.0.0',
          'tensorflow==2.1',
          'tensorflow-addons==0.8.3',
          'numba==0.48.0',
>>>>>>> development
          'scikit-learn',
          'scikit-image',
          'librosa',
          'datetime_glob',
          'matplotlib',
          'tqdm',
          'pint',
          'psutil',
          ],
      python_requires = '>=3.6.0,<3.8.0',
      setup_requires=['pytest-runner','wheel'],
      tests_require=['pytest', ],
      include_package_data=True,
      zip_safe=False)
