from distutils.core import setup
import setuptools
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
  name = 'nested_cv',       
  packages = ['nested_cv'], 
  version = '0.911',     
  license='MIT',       
  description = 'A general package to handle nested cross-validation for any estimator that implements the scikit-learn estimator interface.',   
  author_email = 'ahmedmagdi@outlook.com',
  maintainer_email= 'casperbh.96@gmail.com',
  long_description=long_description,
  long_description_content_type='text/markdown',
  url = 'https://github.com/casperbh96/Nested-Cross-Validation',   
  download_url = 'https://github.com/user/reponame/archive/v_01.tar.gz',    
  keywords = ['ml', 'xgboost', 'numpy','scikit-learn','pandas'],  
  install_requires=[        
          'pandas',
          'matplotlib',
          'sklearn',
          'numpy',
      ],
  classifiers=[
    'Development Status :: 4 - Beta',      
    'Intended Audience :: Developers',   
    'Topic :: Software Development :: Libraries',
    'License :: OSI Approved :: MIT License', 
    'Programming Language :: Python :: 3',     
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
  ],
)
