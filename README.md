#DEAPMaLeLAr
Machine Learning algorithms for DEAP-3600 and LAr experiments

This project is intended to be the space for sharing the ML codes developed by the CIEMAT-DM group.

In this first stage, the main lines (or subprojects) are three: Position Reconstruction Background (neck or surface events) discrimination. Pulse Shape Discrimination

Please, make sure that your contributions are operative and properly documented before commiting to the repo. A very important piece of information is the format in which the input files are expected to be (number of columns and what each column is) so that results are replicable.

In case of doubt or need of more info, contact me: vicente.pesudo@ciemat.es

USEFUL git COMMANDS

To clone the sources from the last release on the repository: git clone https://github.com/vpesudo/deapmalelar.git

This command will download the last version on the server, will create a folder called deapmalelar and will track modification of deapmalelar on the server. It creates a local version of the git server code.

To know the status of your local version (uncommitted files, unaided files, ….): git status

To update your local version from the server one (be careful with unmarked files): git pull

To add a file or a folder in the git repository (you cannot commit a file which isn’t added on the repo): git add path/to/my/file.ext/or/my/folder

To commit all your modifications on your local version (be very careful !!!): git commit -a -m 'your lovely commit message'

To commit a specific file: git commit -m 'your lovely commit message' path/to/my/file.ext

To propagate your commit on the server: git push

To restore a file (erased or modified): git checkout path/to/my/file.ext

After a commit, you always need to push your modifications !!!
