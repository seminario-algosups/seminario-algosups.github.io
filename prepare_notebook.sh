notebook=$1

jupyter nbconvert --clear-output --inplace $notebook
jupyter nbconvert --to markdown $notebook