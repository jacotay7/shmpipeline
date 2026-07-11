# Security policy

## Supported versions

Security fixes are made on the latest released minor version. Older releases
may receive a fix when the maintainer considers a backport low risk.

## Reporting a vulnerability

Do not open a public issue for a vulnerability before coordinated disclosure.
Use GitHub's private vulnerability reporting feature on the repository's
Security tab. If that feature is unavailable, contact the maintainer through
the email address associated with the `Jacob Taylor` commits.

Include a minimal reproduction, affected versions/platforms, impact, and any
known mitigation. A disclosure date will be coordinated after a fix or
mitigation is available.

Shared-memory names and the optional HTTP control plane are not an
authorization boundary. Configure control-plane tokens and bind the service
to a trusted interface; shared-memory payloads are accessible to processes
with OS permission to open the segments.
