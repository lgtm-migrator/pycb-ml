import os
from setuptools import setup

with open("README.md", "r", encoding="utf-8") as f:
    LONG_DESCRIPTION = f.read()

# get github workflow env vars
try:
    version = (os.environ['GIT_TAG_NAME']).replace('v', '')
except KeyError:
    print('Defaulting to 0.0.0')
    version = '0.0.0'

setup(
    name='pycb-ml',
    version=version,
    description='py-clash-bot machine learning',
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    keywords='ml python clash royale',
    author='Martin Miglio',
    url='https://github.com/marmig0404/pycb-ml',
    download_url='https://github.com/marmig0404/pycb-ml/releases',
    install_requires=['keras', 'tensorflow', 'pysimplegui',
                      'pillow', 'pandas', 'plotly', 'ipython',
                      'tqdm', 'scipy', 'sklearn', 'tabulate'],
    packages=['pycbml'],
    python_requires='>=3',
    zip_safe=False,
    classifiers=[
        'Programming Language :: Python :: 3.10',
    ],
)
