# Entry point — dispatches to the active experiment module.
import argparse

parser = argparse.ArgumentParser(description="Run an ML experiment.")
parser.add_argument("experiment", choices=["titanic", "mnist"], default="titanic", nargs="?")
args = parser.parse_args()

if args.experiment == "mnist":
    from mnist.main import main
else:
    from titanic.main import main

if __name__ == "__main__":
    main()
