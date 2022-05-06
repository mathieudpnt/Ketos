from setuptools import setup, find_packages
from ketos.__init__ import __version__

# create distribution and upload to pypi.org with:
#   $ python setup.py sdist bdist_wheel
#   $ twine upload dist/*

setup(name='ketos',
      version=__version__,
      description="MERIDIAN Python package for deep-learning based acoustic detectors and classifiers",
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
          'setuptools>=41.0.0',
          'tensorflow>=2.2,<=2.8',
          'scikit-learn',
          'scikit-image',
          'librosa',
          'datetime_glob',
          'matplotlib',
          'tqdm',
          'pint',
          'psutil',
          'version-parser',
          ],
      python_requires = '>=3.6.0,<=3.9.12',
      setup_requires=['pytest-runner','wheel'],
      tests_require=['pytest', ],
      include_package_data=True,
      zip_safe=False)
