#!/bin/bash -e

# This script contains tests we suggest developers run before pushing commits 
# to the GitHub. Although these tests is contained in the workflow that the 
# CI system will execute, we hope developers can realize errors as early as 
# possible. If developers forget to check these pre-submit tests, The CI system
# is the last line in detecting and notifying developers about test failure. 

# Test 1: python format on syntax errors or undefined names 
echo "Checking python format on syntax errors or undefined names..."
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
echo -e "\033[0;32mTests Passed \033[0m"
echo ""

# Test 2: python format on suggested coding styles
echo "Checking coding styles with the help of flake8..."
flake8 . --count --max-line-length=80 --statistics \
    --ignore=W291,W293,W391,W503 
echo -e "\033[0;32mTests Passed \033[0m"
echo ""

# The warnings we ignored are detailed as follows:
# W291: trailing whitespace
# W293: blank line contains whitespace
# W391: blank line at end of file
# W503: linke break before binary operator 

# Test 3: python unittests
# We currently launch 20 processes with a timeout 900s.
# More processes and a longr timeout can be specified in the future 
echo "Running unittests with the coverage report..."
nosetests -v --processes=20 --process-timeout=900 \
    --with-coverage --cover-package=backend,simulator,program,config,util  
echo -e "\033[0;32mTests Passed \033[0m"
echo ""

echo -e "\033[0;32mAll pre-submit tests are passed! \033[0m"
