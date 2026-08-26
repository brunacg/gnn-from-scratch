# Fix for `ModuleNotFoundError: No module named 'gnn.data'`

## Cause

The repository `.gitignore` contained:

```gitignore
data/
```

In Git, that pattern ignores **every directory named `data`**, including the Python
package directory:

```text
gnn/data/
```

So `gnn/data/__init__.py` and `gnn/data/cora.py` existed locally, which is why the
tests passed on the local machine, but they were not committed to GitHub. The CI
runner therefore checked out a repository with no `gnn.data` package.

## Fix

1. Replace `.gitignore` with the version in this package.
2. Copy the included `gnn/data/` folder into the repository.
3. From the repository root, run:

```bash
git add .gitignore gnn/data/__init__.py gnn/data/cora.py
git status
git commit -m "Fix tracking of gnn.data package"
git push
```

The corrected ignore rule is:

```gitignore
/data/
```

The leading slash means only a **root-level dataset/cache directory** called `data/`
is ignored. `gnn/data/` remains tracked as Python source code.

## Quick verification before committing

Run:

```bash
git check-ignore -v gnn/data/cora.py
```

It should print **nothing**.

Then run:

```bash
git ls-files gnn/data
```

After `git add`, it should list:

```text
gnn/data/__init__.py
gnn/data/cora.py
```
