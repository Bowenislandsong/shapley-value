from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='shapley-value',
    version='0.0.3',
    description='Shapley Value Calculator',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Bowen Song',
    url='https://bowenislandsong.github.io/#/personal',
    project_urls={                 # <-- Project link
        'Source': 'https://github.com/Bowenislandsong/shapley-value',
    },
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
    ],
    python_requires='>=3.7',
)


# python setup.py sdist bdist_wheel
# twine upload dist/* --skip-existing
