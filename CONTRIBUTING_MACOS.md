## Mac 用户的其他注意事项

设置开发环境的基本说明位于 [CONTRIBUTING.md](https://github.com/onyx-dot-app/onyx/blob/main/CONTRIBUTING.md)。

### 设置 Python

确保已经设置好 [Homebrew](https://brew.sh/)。

然后安装 python 3.11。

```bash
brew install python@3.11
```

Add python 3.11 to your path: add the following line to ~/.zshrc

```
export PATH="$(brew --prefix)/opt/python@3.11/libexec/bin:$PATH"
```

> **Note:**
> You will need to open a new terminal for the path change above to take effect.

### Setting up Docker

On macOS, you will need to install [Docker Desktop](https://www.docker.com/products/docker-desktop/) and
ensure it is running before continuing with the docker commands.

### Formatting and Linting

MacOS will likely require you to remove some quarantine attributes on some of the hooks for them to execute properly.
After installing pre-commit, run the following command:

```bash
sudo xattr -r -d com.apple.quarantine ~/.cache/pre-commit
```
