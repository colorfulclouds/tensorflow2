#!/bin/bash

commit_str=$1

if $# == 0
	then
	commit_str="add"
fi
 

git add .
git commit -m "$commit_str"
git push origin master
