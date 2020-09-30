# Release Process

## 0. Pre-Release Checklist

Before starting the release process, verify the following:

* All work required for this release has been completed and the team is ready to release.
* [All CircleCI tests are green on main](https://app.circleci.com/pipelines/github/FeatureLabs/woodwork?branch=main).
* The [ReadtheDocs build](https://readthedocs.com/projects/feature-labs-inc-datatables/) for "latest" is marked as passed. To avoid mysterious errors, best practice is to empty your browser cache when reading new versions of the docs!
* The [public documentation for the "latest" branch](https://feature-labs-inc-datatables.readthedocs-hosted.com/en/latest/) looks correct, and the [changelog](https://feature-labs-inc-datatables.readthedocs-hosted.com/en/latest/changelog.html) includes the last change which was made on `main`.
* Get agreement on the version number to use for the release.

#### Version Numbering

Woodwork uses [semantic versioning](https://semver.org/). Every release has a major, minor and patch version number, and are displayed like so: `<majorVersion>.<minorVersion>.<patchVersion>`.

If you'd like to create a development release, which won't be deployed to pypi and conda and marked as a generally-available production release, please add a "dev" prefix to the patch version, i.e. `X.X.devX`. Note this claims the patch number--if the previous release was `0.12.0`, a subsequent dev release would be `0.12.dev1`, and the following release would be `0.12.2`, *not* `0.12.1`. Development releases deploy to [test.pypi.org](https://test.pypi.org/project/woodwork/) instead of to [pypi.org](https://pypi.org/project/woodwork).

## 1. Create Woodwork release on Github

#### Create release branch

1. Branch off of Woodwork `main` and name the branch the release version number (e.g. v0.13.3)

#### Bump version number

1. Bump `__version__` in `setup.py`, `woodwork/version.py`, and `woodwork/tests/test_version.py`.

#### Update changelog

1. Replace "Future Release" in `docs/source/changelog.rst` with the current date

    ```
    **v0.13.3 September 28, 2020**
    ```

2. Remove any unused changelog sections for this release (e.g. Fixes, Testing Changes)
3. Add yourself to the list of contributors to this release and **put the contributors in alphabetical order**
4. The release PR does not need to be mentioned in the list of changes
5. Add a commented out "Future Release" section with all of the changelog sections above the current section

    ```
    .. **Future Release**
        * Enhancements
        * Fixes
        * Changes
        * Documentation Changes
        * Testing Changes

    .. Thanks to the following people for contributing to this release:
    ```

An example can be found here: <https://github.com/FeatureLabs/woodwork/pull/158>

Checklist before merging:

* All tests are currently green on checkin and on `main`.
* The ReadtheDocs build for the release PR branch has passed, and the resulting docs contain the expected release notes.
* PR has been reviewed and approved.
* Confirm with the team that `main` will be frozen until step 2 ([Github Release](2-create-github-release)) is complete.

After merging, verify again that ReadtheDocs "latest" is correct.

## 2. Create Github Release

After the release pull request has been merged into the `main` branch, it is time draft the github release. [Example release](https://github.com/FeatureLabs/woodwork/releases/tag/v0.0.2)

* The target should be the `main` branch
* The tag should be the version number with a v prefix (e.g. v0.13.3)
* Release title is the same as the tag
* Release description should be the full changelog updates for the release, including the line thanking contributors.  Contributors should also have their links changed from the docs syntax (:user:\`gsheni\`) to github syntax (@gsheni)
* This is not a pre-release
* Publishing the release will automatically upload the package to PyPI
