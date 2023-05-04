REGEX="^(fix|feat|build|chore|ci|docs|refactor|perf|style|test|deploy)(\((\w+)\))?: .*"
ERROR_MSG="Commit message format must match regex \"${REGEX}\""
if [[ $1 =~ $REGEX ]]; then
    echo "Nice commit!"
else
    echo "Bad commit \"$1\", check format."
    echo $ERROR_MSG
    exit 1
fi
