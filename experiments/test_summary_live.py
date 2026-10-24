"""Legacy entry point. Importing this module never makes API requests.

Run `python -m experiments.summary_pilot --live` for the five-case pilot.
"""

if __name__ == "__main__":
    from experiments.summary_pilot import main

    raise SystemExit(main())
