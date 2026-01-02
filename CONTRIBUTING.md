# Contributing to Contact Center Reasoning Model

Thank you for your interest in contributing! This document provides guidelines for contributing to this project.

## Code of Conduct

Be respectful, inclusive, and professional in all interactions.

## How to Contribute

### Reporting Bugs

1. **Check existing issues** to avoid duplicates
2. **Use the bug report template** when creating an issue
3. **Include**:
   - Clear description of the problem
   - Steps to reproduce
   - Expected vs. actual behavior
   - Environment details (OS, Python version, hardware)
   - Error messages and logs

### Suggesting Enhancements

1. **Use the feature request template**
2. **Describe the use case** and benefits
3. **Provide examples** of how it would work
4. **Consider backwards compatibility**

### Pull Requests

1. **Fork the repository**
2. **Create a feature branch** from `main`
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes**
   - Follow the coding standards below
   - Add tests if applicable
   - Update documentation
4. **Commit with clear messages**
   ```bash
   git commit -m "Add feature: brief description"
   ```
5. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```
6. **Open a Pull Request**
   - Use the PR template
   - Link related issues
   - Describe changes and testing performed

## Coding Standards

### Python Style

- Follow **PEP 8** style guide
- Use **Black** for formatting (line length: 100)
  ```bash
  black scripts/
  ```
- Use **type hints** for function signatures
  ```python
  def analyze_transcript(transcript: str, model: str = "mistral") -> Dict[str, Any]:
      pass
  ```
- Write **docstrings** for all public functions
  ```python
  def process_data(data: List[Dict]) -> List[Dict]:
      """Process raw data into training format.

      Args:
          data: List of raw examples

      Returns:
          List of processed training examples
      """
      pass
  ```

### Code Organization

- **One class/function per file** when possible
- **Keep functions small** (< 50 lines ideally)
- **Use descriptive variable names**
  - Good: `quality_score`, `agent_performance`
  - Bad: `qs`, `ap`, `x`

### Testing

- Write tests for new features
- Maintain >80% code coverage
- Use `pytest` for testing
  ```bash
  pytest tests/ --cov=scripts/
  ```

### Documentation

- Update `README.md` for user-facing changes
- Add docstrings to all new functions/classes
- Include usage examples in doc comments
- Update `docs/` for architectural changes

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR-USERNAME/contact-center-reasoning.git
cd contact-center-reasoning

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dev dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install
```

## Testing Your Changes

```bash
# Run unit tests
pytest tests/

# Run integration tests
pytest tests/integration/

# Check code formatting
black --check scripts/

# Run linter
flake8 scripts/

# Type checking
mypy scripts/
```

## Commit Message Guidelines

Use clear, descriptive commit messages:

- **feat**: New feature
- **fix**: Bug fix
- **docs**: Documentation changes
- **style**: Formatting, missing semicolons, etc.
- **refactor**: Code restructuring
- **test**: Adding tests
- **chore**: Maintenance tasks

Examples:
```
feat: Add multilingual support for Spanish transcripts
fix: Resolve agent fingerprinting clustering bug
docs: Update RAG integration examples
refactor: Simplify temporal feature extraction
test: Add unit tests for reasoning validation
```

## Pull Request Process

1. **Ensure all tests pass**
2. **Update documentation** if needed
3. **Add yourself to CONTRIBUTORS.md** (if first contribution)
4. **Request review** from maintainers
5. **Address review comments** promptly
6. **Squash commits** if requested
7. **Maintainers will merge** once approved

## Areas for Contribution

### High Priority

- [ ] Multi-language support (Spanish, French, Mandarin)
- [ ] Real-time inference API with FastAPI
- [ ] Automated coaching email templates
- [ ] Performance benchmarking suite
- [ ] Integration with Genesys Cloud

### Medium Priority

- [ ] Jupyter notebooks with examples
- [ ] Docker deployment configuration
- [ ] Model compression techniques
- [ ] Alternative base models (Phi, Gemma)
- [ ] Streaming inference support

### Documentation

- [ ] API reference documentation
- [ ] Video tutorials
- [ ] Use case examples
- [ ] Best practices guide
- [ ] Troubleshooting FAQ

### Testing

- [ ] Unit test coverage >90%
- [ ] Integration tests for all scripts
- [ ] Performance regression tests
- [ ] Edge case testing

## Questions?

- **Open a discussion** in GitHub Discussions
- **Join our community** (Discord/Slack link)
- **Email maintainers** at your.email@example.com

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes for significant contributions
- Invited to join the core team for sustained contributions

Thank you for making this project better!
