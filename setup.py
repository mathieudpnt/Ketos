from setuptools import setup, find_packages

# create distribution and upload to pypi.org with:
#   $ python setup.py sdist bdist_wheel
#   $ twine upload dist/*

setup(name='ketos',
      version='2.0.0-beta',
      description="MERIDIAN Python package for deep-learning based acoustic detector and classifiers",
      url='https://gitlab.meridian.cs.dal.ca/public_projects/ketos',
      author='Fabio Frazao, Oliver Kirsebom',
      author_email='fsfrazao@dal.ca, oliver.kirsebom@dal.ca',
      license='GNU General Public License v3.0',
      packages=find_packages(),
      install_requires=[
          'numpy',
          'tables',
          'scipy',
          'pandas',
          'tensorflow==2.1',
          'scikit-learn',
          'scikit-image',
          'librosa',
          'datetime_glob',
          'matplotlib',
          'tqdm',
          'pint',
          'psutil',
          ],
      setup_requires=['pytest-runner','wheel'],
      tests_require=['pytest', ],
      include_package_data=True,
      zip_safe=False)
