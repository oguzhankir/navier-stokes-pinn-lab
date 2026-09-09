# Contributor guide

- Keep the project a small, reproducible 2D Navier–Stokes lab, not a general PINN framework.
- Use English for code, documentation, identifiers and commit messages.
- Preserve the MIT license and existing user changes. Work on `oguzhankir/<topic>` branches; use DCO sign-off.
- The showcase must come from an actually trained checkpoint. Never substitute an exact field or numerical solution for a PINN prediction.
- Forward training may use the prescribed initial/boundary conditions, not future reference labels. Evaluation must use independent points.
- Use density-normalized pressure, periodic spatial domain `[-pi, pi)^2`, and explicit viscosity/time units. Align pressure gauges separately at each evaluation time.
- Reference formulas, numerical solvers and neural predictions are different sources; label them explicitly. A numerical solver must genuinely integrate the PDE.
- No finite sampled residual or visualization proves global regularity, singularity formation, or a Millennium Problem result. High Reynolds number does not make the exact 2D Taylor–Green benchmark turbulent.
- Report measured error and separate training, evaluation, numerical integration and rendering times. Do not invent performance, accuracy, or GPU figures.
- Tests should cover exact residuals, derivatives, periodicity, initial conditions, pressure gauge, numerical convergence, checkpoint replay, and CLI smoke behavior.
- Keep explanatory documentation in README.md. Add focused tests and code, not a docs tree or speculative roadmap.
- Do not run paid cloud jobs, publish hosting, change repository visibility, merge PRs, or bypass permissions without explicit authorization.
