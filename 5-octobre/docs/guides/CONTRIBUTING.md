# Contributing Guide

## Development Setup

1. **Environment Setup**
   - Use Python 3.8+
   - Create a virtual environment
   - Install development dependencies: `pip install -r requirements-dev.txt`

2. **Code Style**
   - We use Black for code formatting
   - Type hints are required for all new code
   - Follow PEP 8 guidelines
   - Document all functions and classes

3. **Git Workflow**
   - Branch naming: `feature/`, `bugfix/`, `hotfix/` prefixes
   - Commit messages should be clear and descriptive
   - Keep commits atomic and focused
   - Rebase your branch before submitting PR

4. **Testing**
   - Write unit tests for new features
   - Maintain test coverage above 80%
   - Run the full test suite before submitting PR
   - Include both positive and negative test cases

5. **Documentation**
   - Update relevant documentation
   - Include docstrings for new functions/classes
   - Update README if needed
   - Add comments for complex logic

6. **Pull Request Process**
   - Create a feature branch from `develop`
   - Keep PRs focused and reasonable in size
   - Fill out the PR template completely
   - Request review from relevant team members
   - Address review comments promptly

7. **Code Review**
   - Review PRs within 48 hours
   - Be constructive and respectful
   - Focus on code quality and maintainability
   - Test the changes locally if needed

## Project Structure

- Place new features in appropriate modules
- Follow existing patterns for consistency
- Create new modules only when necessary
- Keep files focused and manageable in size

## Best Practices

1. **Code Quality**
   - Write self-documenting code
   - Follow DRY principles
   - Use meaningful variable names
   - Keep functions small and focused

2. **Performance**
   - Consider memory usage
   - Optimize database queries
   - Use appropriate data structures
   - Profile code when necessary

3. **Security**
   - Never commit sensitive data
   - Use environment variables for secrets
   - Follow security best practices
   - Report security issues privately

4. **Dependencies**
   - Pin dependency versions
   - Regular dependency updates
   - Minimize new dependencies
   - Document why dependencies are needed
