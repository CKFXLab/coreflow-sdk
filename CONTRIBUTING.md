# Contributing to COREFLOW SDK

Thank you for considering contributing to COREFLOW SDK! We welcome contributions from everyone.

## Table of Contents
1. [Reporting Issues](#reporting-issues)  
2. [Development Setup](#development-setup)  
3. [Running Tests](#running-tests)  
4. [Style Guidelines](#style-guidelines)  
5. [Pull Request Process](#pull-request-process)  

## Reporting Issues
Use our [bug report template](.github/ISSUE_TEMPLATE/bug_report.md) when opening a new issue.

## Development Setup
1. Fork the repository on GitHub  
1. Clone your fork:
```bash
git clone https://github.com/CKFXLab/coreflow-sdk.git
cd coreflow-sdk
```
1. Install dependencies
```bash
python -m pip install -e .[dev]
```

## Running Tests
we use ```pytest```to run all tests:
```bash
pytest
```

## Style Guidelines
* **Formater**: [Black](https://github.com/psf/black)
* **Linter**: [Flake8](https://flake8.pycqa.org/en/latest/)

Run formatting and linting before committing:
```bash
black {source_file_or_directory}
```

## Pull Request Process
1. Create a new branch: git checkout -b feature/your-feature-name

1. Commit your changes using [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/)

1. Push to your fork and open a PR against main

1. Ensure all CI checks pass

1. Request a review from maintainers

We appreciate your contributions and will review them as quickly as possible!