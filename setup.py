from os import path

from setuptools import find_packages, setup

dirname = path.abspath(path.dirname(__file__))
with open(path.join(dirname, 'README.md')) as f:
    long_description = f.read()

extras_require = {'dask': open('dask-requirements.txt').readlines(),
                  'koalas': open('koalas-requirements.txt').readlines()}
extras_require['complete'] = sorted(set(sum(extras_require.values(), [])))

setup(
    name='woodwork',
    author='Alteryx, Inc.',
    author_email='support@featurelabs.com',
    license='BSD 3-clause',
    version='0.0.8',
    description='a two-dimensional data object with labeled axes and typing information',
    url='https://github.com/alteryx/woodwork/',
    classifiers=[
         'Development Status :: 3 - Alpha',
         'Intended Audience :: Developers',
         'Programming Language :: Python :: 3',
         'Programming Language :: Python :: 3.6',
         'Programming Language :: Python :: 3.7',
         'Programming Language :: Python :: 3.8'
    ],
    packages=find_packages(),
    install_requires=open('requirements.txt').readlines(),
    tests_require=open('test-requirements.txt').readlines(),
    python_requires='>=3.6, <4',
    extras_require=extras_require,
    keywords='data science machine learning typing',
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'woodwork = woodwork.__main__:cli'
        ]
    },
    long_description=long_description,
    long_description_content_type='text/markdown'
)
