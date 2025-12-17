Contributing
============

Thank you for your interest in contributing to Sionna DL 6G Demos!

Development Setup
-----------------

1. Fork and clone the repository
2. Install development dependencies:

   .. code-block:: bash

      poetry install --with dev

3. Install pre-commit hooks:

   .. code-block:: bash

      pre-commit install

Code Style
----------

This project uses:

- **Black** for code formatting (line length: 88)
- **Flake8** for linting
- **autoflake** for removing unused imports

Run formatting before committing:

.. code-block:: bash

   black .
   flake8 .

Running Tests
-------------

.. code-block:: bash

   pytest tests/

Building Documentation
----------------------

.. code-block:: bash

   cd docs
   make html

The built documentation will be in ``docs/_build/html/``.

Pull Request Process
--------------------

1. Create a feature branch from ``main``
2. Make your changes with clear commit messages
3. Ensure tests pass and code is formatted
4. Submit a pull request with a clear description

Reporting Issues
----------------

Please use GitHub Issues to report bugs or request features. Include:

- Clear description of the problem
- Steps to reproduce
- Expected vs. actual behavior
- Environment details (Python version, GPU, etc.)
