from setuptools import setup, find_packages
import os

# Read README for long description
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

# Read version from package
def get_version():
    """Get version from package __init__.py"""
    init_path = os.path.join('shapley_value', '__init__.py')
    if os.path.exists(init_path):
        with open(init_path, 'r') as f:
            for line in f:
                if line.startswith('__version__'):
                    return line.split('=')[1].strip().strip('\'"')
    return '0.0.4'  # Fallback version

setup(
    name='shapley-value',
    version=get_version(),
    description='A comprehensive Python package for calculating Shapley values in cooperative game theory',
    long_description=long_description,
    long_description_content_type='text/markdown',
    
    # Author and contact information
    author='Bowen Song',
    author_email='bowenson@example.com',  # Add your email if you want
    maintainer='Bowen Song',
    
    # URLs and project links
    url='https://github.com/Bowenislandsong/shapley-value',
    project_urls={
        'Homepage': 'https://github.com/Bowenislandsong/shapley-value',
        'Source': 'https://github.com/Bowenislandsong/shapley-value',
        'Bug Reports': 'https://github.com/Bowenislandsong/shapley-value/issues',
        'Documentation': 'https://github.com/Bowenislandsong/shapley-value#readme',
        'Examples': 'https://github.com/Bowenislandsong/shapley-value/tree/main/examples',
        'Author': 'https://bowenislandsong.github.io/#/personal',
    },
    
    # License
    license='MIT',
    
    # Package configuration
    packages=find_packages(exclude=['tests*', 'examples*']),
    include_package_data=True,
    zip_safe=False,
    
    # Dependencies
    install_requires=[
        'joblib>=1.0.0',  # For parallel processing
        'pandas>=1.0.0',  # For data export functionality
    ],
    extras_require={
        'dev': [
            'pytest>=6.0',
            'pytest-cov',
            'black',
            'flake8',
            'mypy',
        ],
        'examples': [
            'pandas>=1.0.0',  # For data export in examples
            'numpy>=1.18.0',  # For advanced ML examples
        ],
        'performance': [
            'pandas>=1.0.0',
            'psutil',  # For memory monitoring
        ],
    },
    
    # Testing
    tests_require=[
        'pytest>=6.0',
        'pytest-cov',
    ],
    test_suite='tests',
    
    # Entry points (if you want CLI commands)
    entry_points={
        'console_scripts': [
            # 'shapley-calc=shapley_value.cli:main',  # Uncomment if you add CLI
        ],
    },
    
    # Package metadata
    keywords=[
        'shapley', 'shapley-value', 'game-theory', 'cooperative-games',
        'fair-allocation', 'coalition', 'marginal-contribution',
        'economics', 'mathematics', 'optimization', 'machine-learning',
        'feature-importance', 'explainable-ai', 'xai'
    ],
    
    # Classifiers for PyPI
    classifiers=[
        # Development Status
        'Development Status :: 5 - Production/Stable',
        
        # Intended Audience
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Education',
        'Intended Audience :: Financial and Insurance Industry',
        
        # Topic
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Office/Business :: Financial',
        
        # License
        'License :: OSI Approved :: MIT License',
        
        # Programming Language
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
        'Programming Language :: Python :: 3 :: Only',
        
        # Operating System
        'Operating System :: OS Independent',
        
        # Environment
        'Environment :: Console',
        'Environment :: Web Environment',
        
        # Natural Language
        'Natural Language :: English',
    ],
    
    # Python version requirement
    python_requires='>=3.7',
    
    # Platform
    platforms=['any'],
)


# python setup.py sdist bdist_wheel
# twine upload dist/* --skip-existing
