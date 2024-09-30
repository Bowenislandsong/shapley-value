
from setuptools import setup, find_packages

with open('README.md', 'r') as f:
    long_description = f.read()

setup(
    name='shapley-value',
    version='0.0.3',
    description='Shapley Value Calculator',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Your Name',
    author_email='your@email.com',
    license='MIT',
    packages=find_packages(),
    install_requires=[],
    tests_require=['unittest'],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Mathematics'
    ]
)

# python setup.py sdist bdist_wheel
# twine upload dist/* --skip-existing