# Contributing to rs-tfhe

Thank you for your interest in contributing to rs-tfhe! This document provides guidelines and information for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contributing Guidelines](#contributing-guidelines)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Release Process](#release-process)

## Code of Conduct

This project adheres to a code of conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to [conduct@your-domain.com](mailto:conduct@your-domain.com).

## Getting Started

### Prerequisites

- Rust 1.70.0 or later
- Git
- Basic understanding of homomorphic encryption concepts

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/your-username/rs-tfhe.git
   cd rs-tfhe
   ```
3. Add the upstream repository:
   ```bash
   git remote add upstream https://github.com/original-org/rs-tfhe.git
   ```

## Development Setup

### 1. Install Dependencies

```bash
# Install Rust (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install additional tools
cargo install cargo-audit cargo-tarpaulin cargo-geiger
```

### 2. Build the Project

```bash
# Build in debug mode
cargo build

# Build in release mode
cargo build --release

# Build with all features
cargo build --all-features
```

### 3. Run Tests

```bash
# Run all tests
cargo test

# Run tests with specific features
cargo test --features "lut-bootstrap"

# Run tests in release mode
cargo test --release
```

### 4. Run Examples

```bash
# Basic examples
cargo run --example add_two_numbers --release
cargo run --example gates_with_strategies --release

# LUT bootstrapping examples
cargo run --example lut_bootstrapping --features "lut-bootstrap" --release
cargo run --example lut_add_two_numbers --features "lut-bootstrap" --release
```

## Contributing Guidelines

### Types of Contributions

We welcome several types of contributions:

- **Bug fixes**: Fix existing issues
- **New features**: Add new functionality
- **Performance improvements**: Optimize existing code
- **Documentation**: Improve or add documentation
- **Tests**: Add or improve test coverage
- **Examples**: Add new usage examples

### Before You Start

1. **Check existing issues**: Look for existing issues or discussions
2. **Create an issue**: For significant changes, create an issue first
3. **Discuss**: For major changes, discuss your approach in the issue

### Branch Naming

Use descriptive branch names:

- `fix/issue-123-description`
- `feature/lut-bootstrap-enhancement`
- `docs/update-readme`
- `perf/optimize-fft`

## Pull Request Process

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Changes

- Write clean, well-documented code
- Add tests for new functionality
- Update documentation as needed
- Follow the coding standards

### 3. Test Your Changes

```bash
# Run all tests
cargo test --all-features

# Run clippy
cargo clippy --all-targets --all-features -- -D warnings

# Check formatting
cargo fmt --all -- --check

# Run security audit
cargo audit
```

### 4. Commit Your Changes

Use conventional commit messages:

```
feat: add new LUT bootstrapping feature
fix: resolve issue with parameter validation
docs: update API documentation
test: add tests for new functionality
```

### 5. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Then create a pull request on GitHub.

## Coding Standards

### Rust Style

- Follow standard Rust formatting (`cargo fmt`)
- Use `cargo clippy` to catch common issues
- Prefer explicit types over type inference in public APIs
- Use meaningful variable and function names

### Code Organization

- Keep functions small and focused
- Use appropriate module organization
- Add documentation for public APIs
- Include examples in documentation

### Error Handling

- Use `Result<T, E>` for fallible operations
- Provide meaningful error messages
- Use appropriate error types

### Performance

- Consider performance implications
- Add benchmarks for performance-critical code
- Use appropriate data structures
- Avoid unnecessary allocations

## Testing

### Test Types

1. **Unit Tests**: Test individual functions and methods
2. **Integration Tests**: Test component interactions
3. **Property Tests**: Test mathematical properties
4. **Performance Tests**: Benchmark critical operations

### Writing Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_function_name() {
        // Arrange
        let input = create_test_input();
        
        // Act
        let result = function_under_test(input);
        
        // Assert
        assert_eq!(result, expected_output);
    }
}
```

### Running Tests

```bash
# Run specific test
cargo test test_function_name

# Run tests with output
cargo test -- --nocapture

# Run tests in a specific module
cargo test module_name
```

## Documentation

### Code Documentation

- Document all public APIs
- Use rustdoc format
- Include examples in documentation
- Explain complex algorithms

```rust
/// Encrypts a boolean value using TLWE.
///
/// # Arguments
/// * `plaintext` - The boolean value to encrypt
/// * `key` - The secret key for encryption
///
/// # Returns
/// An encrypted ciphertext
///
/// # Example
/// ```
/// use rs_tfhe::key;
/// use rs_tfhe::utils::Ciphertext;
///
/// let key = key::SecretKey::new();
/// let encrypted = Ciphertext::encrypt(true, &key.key_lv0);
/// ```
pub fn encrypt(plaintext: bool, key: &SecretKey) -> Ciphertext {
    // Implementation
}
```

### README Updates

- Update README.md for significant changes
- Add new examples
- Update feature lists
- Update installation instructions

## Release Process

### Versioning

We follow semantic versioning (SemVer):

- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Checklist

- [ ] All tests pass
- [ ] Documentation is updated
- [ ] Changelog is updated
- [ ] Version is bumped
- [ ] Release notes are prepared

## Getting Help

### Resources

- **Documentation**: Check the docs/ directory
- **Examples**: Look at examples/ directory
- **Issues**: Search existing issues
- **Discussions**: Use GitHub Discussions

### Contact

- **Email**: [maintainers@your-domain.com](mailto:maintainers@your-domain.com)
- **GitHub Issues**: For bug reports and feature requests
- **GitHub Discussions**: For questions and general discussion

## Recognition

Contributors will be recognized in:

- CONTRIBUTORS.md file
- Release notes
- Project documentation

Thank you for contributing to rs-tfhe!
