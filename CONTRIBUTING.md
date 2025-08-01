<!-- omit in toc -->

# Contributing to optype

First off, thanks for taking the time to contribute! ❤️

All types of contributions are encouraged and valued.
See the [Table of Contents](#table-of-contents) for different ways to help and
details about how this project handles them.
Please make sure to read the relevant section before making your contribution.
It will make it a lot easier for us maintainers and smooth out the experience
for all involved.
The community looks forward to your contributions. 🎉

> [!NOTE]
> And if you like optype, but just don't have time to contribute, that's fine.
> There are other easy ways to support the project and show your appreciation,
> which we would also be very happy about:
>
> - Star the project
> - Tweet about it
> - Refer this project in your project's readme
> - Mention the project at local meetups and tell your friends/colleagues

<!-- omit in toc -->

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [I Have a Question](#i-have-a-question)
- [I Want To Contribute](#i-want-to-contribute)
  - [Reporting Bugs](#reporting-bugs)
  - [Suggesting Enhancements](#suggesting-enhancements)
  - [Your First Code Contribution](#your-first-code-contribution)
  - [Improving The Documentation](#improving-the-documentation)

## Code of Conduct

This project and everyone participating in it is governed by the
[optype Code of Conduct][OP_TENETS].
By participating, you are expected to uphold this code.
Please report unacceptable behavior to `jhammudoglu<at>gmail<dot>com`.

## I Have a Question

> [!NOTE]
> If you want to ask a question, we assume that you have read the
> available [Documentation][OP-README].

Before you ask a question, it is best to search for existing [Issues][OP-ISSUES]
that might help you.
In case you have found a suitable issue and still need clarification,
you can write your question in this issue.
It is also advisable to search the internet for answers first.

If you then still feel the need to ask a question and need clarification, we
recommend the following:

- Open an [Issue][OP-ISSUES].
- Provide as much context as you can about what you're running into.
- Provide project and platform versions (Python, mypy, pyright, ruff, etc),
  depending on what seems relevant.

We will then take care of the issue as soon as possible.

## I Want To Contribute

> ### Legal Notice <!-- omit in toc -->
>
> When contributing to this project,
> you must agree that you have authored 100% of the content,
> that you have the necessary rights to the content and that the content you
> contribute may be provided under the project license.

### Reporting Bugs

<!-- omit in toc -->

#### Before Submitting a Bug Report

A good bug report shouldn't leave others needing to chase you up for more
information.
Therefore, we ask you to investigate carefully, collect information and
describe the issue in detail in your report.
Please complete the following steps in advance to help us fix any potential
bug as fast as possible.

- Make sure that you are using the latest version.
- Determine if your bug is really a bug and not an error on your side e.g.
  using incompatible environment components/versions
  (Make sure that you have read the [documentation][OP-README].
  If you are looking for support, you might want to check
  [this section](#i-have-a-question)).
- To see if other users have experienced (and potentially already solved)
  the same issue you are having, check if there is not already a bug report
  existing for your bug or error in the [bug tracker][OP-ISSUES].
- Also make sure to search the internet (including Stack Overflow) to see if
  users outside of the GitHub community have discussed the issue.
- Collect information about the bug:
  - Stack trace (Traceback)
  - OS, Platform and Version (Windows, Linux, macOS, x86, ARM)
  - Version of the interpreter, compiler, SDK, runtime environment,
    package manager, depending on what seems relevant.
  - Possibly your input and the output
  - Can you reliably reproduce the issue?
    And can you also reproduce it with older versions?

<!-- omit in toc -->

#### How Do I Submit a Good Bug Report?

> You must never report security related issues, vulnerabilities or bugs
> including sensitive information to the issue tracker, or elsewhere in public.
> Instead sensitive bugs must be sent by email to `jhammudoglu<at>gmail<dot>com`.

We use GitHub issues to track bugs and errors.
If you run into an issue with the project:

- Open an [Issue][OP-ISSUES].
  (Since we can't be sure at this point whether it is a bug or not,
  we ask you not to talk about a bug yet and not to label the issue.)
- Explain the behavior you would expect and the actual behavior.
- Please provide as much context as possible and describe the
  *reproduction steps* that someone else can follow to recreate the issue on
  their own.
  This usually includes your code.
  For good bug reports you should isolate the problem and create a reduced test
  case.
- Provide the information you collected in the previous section.

Once it's filed:

- The project team will label the issue accordingly.
- A team member will try to reproduce the issue with your provided steps.
  If there are no reproduction steps or no obvious way to reproduce the issue,
  the team will ask you for those steps and mark the issue as `needs-repro`.
  Bugs with the `needs-repro` tag will not be addressed until they are
  reproduced.
- If the team is able to reproduce the issue, it will be marked `needs-fix`,
  as well as possibly other tags (such as `critical`), and the issue will be
  left to be [implemented by someone](#your-first-code-contribution).

### Suggesting Enhancements

This section guides you through submitting an enhancement suggestion for
optype, **including completely new features and minor improvements to existing
functionality**.
Following these guidelines will help maintainers and the community to
understand your suggestion and find related suggestions.

<!-- omit in toc -->

#### Before Submitting an Enhancement

- Make sure that you are using the latest version.
- Read the [documentation][OP-README] carefully and find out if the functionality is
  already covered, maybe by an individual configuration.
- Perform a [search][OP-ISSUES] to see if the enhancement has already been suggested.
  If it has, add a comment to the existing issue instead of opening a new one.
- Find out whether your idea fits with the scope and aims of the project.
  It's up to you to make a strong case to convince the project's developers of
  the merits of this feature. Keep in mind that we want features that will be
  useful to the majority of our users and not just a small subset. If you're
  just targeting a minority of users, consider writing an add-on/plugin library.

<!-- omit in toc -->

#### How Do I Submit a Good Enhancement Suggestion?

Enhancement suggestions are tracked as [GitHub issues][OP-ISSUES].

- Use a **clear and descriptive title** for the issue to identify the
  suggestion.
- Provide a **step-by-step description of the suggested enhancement** in as
  many details as possible.
- **Describe the current behavior** and **explain which behavior you expected
  to see instead** and why. At this point you can also tell which alternatives
  do not work for you.
- **Explain why this enhancement would be useful** to most optype users.
  You may also want to point out the other projects that solved it better and
  which could serve as inspiration.

### Your First Code Contribution

Ensure you have [`uv`][GH-UV] installed. Now you can install the dev dependencies:

```bash
uv sync
```

This will install all the dependencies needed to run the linters, type-checkers,
and unit tests.

#### Lefthook

[Lefthook](https://github.com/evilmartians/lefthook) is a modern Git hooks manager,
which automatically lints and formats your code before committing it, which helps
avoid CI failures.

To install lefthook as a `uv` tool ([docs](https://docs.astral.sh/uv/concepts/tools/)),
run:

```bash
$ uv tool install lefthook --upgrade
Resolved 1 package in 139ms
Audited 1 package in 0.00ms
Installed 1 executable: lefthook
```

Now the git hooks can be installed by running:

```bash
$ uvx lefthook install
sync hooks: ✔️ (post-checkout, post-merge, pre-commit)
```

To see if everything is set up correctly, you can run the validation command:

```bash
$ uvx lefthook validate
All good
```

See <https://lefthook.dev/> for more information.

#### Tox

The linters, type-checkers, and unit tests can easily be run with [`tox`][GH-TOX]:

```bash
uvx tox p
```

This will run `pytest` in parallel on all supported Python versions, as well as the
linters (`dprint`, `ruff`, and `repo-review`) and the type-checkers (`mypy` and
`basedpyright`).

### Improving The Documentation

All [documentation] lives in the `README.md`. Please read it carefully before
proposing any changes. Ensure that the markdown is formatted correctly with
[`dprint`](https://dprint.dev/) by running `uv run dprint fmt`.

<!-- omit in toc -->

## Attribution

This guide is based on the **contributing-gen**.
[Make your own](https://github.com/bttger/contributing-gen)!

[OP-ISSUES]: https://github.com/jorenham/optype/issues
[OP-README]: https://github.com/jorenham/optype/blob/master/README.md#optype
[OP_TENETS]: https://github.com/jorenham/optype/blob/master/CODE_OF_CONDUCT.md
[GH-TOX]: https://github.com/tox-dev/tox
[GH-UV]: https://github.com/astral-sh/uv
