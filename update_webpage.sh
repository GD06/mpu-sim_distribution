if [ "$#" -ne 1 ]
then 
    echo "Usage: ./update_webpage.sh [webpage_repo_dir] "
    exit 1
fi

COVERAGE_DIR=$1/coverage 
echo $COVERAGE_DIR

nosetests -v --processes=20 --process-timeout=900 --with-coverage \
    --cover-package=backend,simulator,program,config,util --cover-html \
    --cover-html-dir=${COVERAGE_DIR}

echo "Finished generating coverage report"

DOC_DIR=$1/api_doc 
pdoc3 --html -o ${DOC_DIR} --force backend 
pdoc3 --html -o ${DOC_DIR} --force simulator 
pdoc3 --html -o ${DOC_DIR} --force program 
pdoc3 --html -o ${DOC_DIR} --force config 
pdoc3 --html -o ${DOC_DIR} --force util 

echo "Finishd generating API documentation"
