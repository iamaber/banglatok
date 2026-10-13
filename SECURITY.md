# Security reporting

Report vulnerabilities privately to **aber.islam.dev@gmail.com**, or use the repository's private vulnerability reporting feature if enabled. Include the affected version, OS/architecture, a minimal reproduction and expected impact. Do not post credentials, private corpus text or exploit details in a public issue before coordination.

This project is in beta. Security fixes target current development and the latest published release; there is no supported older-release series yet. Prefer maintained Python and Go versions for deployment and source builds.

The native library runs inside the Python process. Report native crashes, memory-safety problems, unsafe archive/build behavior and unexpected network activity. Model loading uses JSON; the package does not automatically fetch models or execute model-supplied Python code.

Contributors should keep build/publish jobs separate, retain least-privilege workflow permissions and preserve third-party notices. Only the protected publishing job receives an OIDC token permission; ordinary pushes and pull requests do not publish packages.
