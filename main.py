# Entry point — dispatches to the active experiment module.
import sys

if len(sys.argv) > 1 and sys.argv[1] == "mnist":
    from mnist.main import main
else:
    from titanic.main import main

if __name__ == "__main__":
    main()
