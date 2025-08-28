# Useful Notes

This note contains some useful things to remember when building docker images

- If there are some files that cannot be found during build, even if its rare, it's best to check the .dockignore file
- If . is used as the context, it will be from the location which the .yml build file is run
