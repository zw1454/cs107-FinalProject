from distutils.core import setup

setup(
  name = 'zapnAD',         
  packages = ['zapnAD'],
  version = '1.0',
  license='GPL-3.0',
  description = 'A package for automatic differentiation', 
  url = 'https://github.com/cs107-zapn/cs107-FinalProject',
  download_url = 'https://github.com/cs107-zapn/cs107-FinalProject/archive/refs/tags/v.03.tar.gz',    # I explain this later on
  keywords = ['optimization', 'autodifferntiation'], 
  install_requires=[         
          'numpy'
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',
    'Programming Language :: Python :: 3',      
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
  ],
)
