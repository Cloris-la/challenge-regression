# Feedback
- The main branch is not clean, there are ipynb with unclear names all over the place, copies of readme and unclear names .md
- there is a "hanieh" folder? branches exist for that :) 
- the main branch is meant to be the final project but the architecture is not easy to follow/read/maintain
- it's good practice to organize your code in logical units like:
  - data_preprocessing.py
  - data_viz.py
  - models_scripts/linear_regression.py
  - models_scripts/catboost.py
  - models_files/catboost.cmb
- Names of other branches are not clear: hanieh, fang, mengstu, estefania would have been better
- I don't see a lot of commits or merge in the working tree of this main branch. It's good enough but try creating logical and regular commits/merge to keep track of the changes and versions.
- code is not clean:
  - notebook instead of .py files
  - no documentation, dynamic typing or even functions. Focus on OOP for the next projects (using .py files)

Overall this project is a bit messy, I can't reuse the code easily to run it and test on my side. You have results so you've accomplished it but there is room for improvement. You can focus on this in the future:
- make sure the readme is clean and complete without format issues
- use .py file, docstring, dynamic typing, OOP
- manage the github repo properly
- build a clear and readable code architecture to help people understand what you've done

This is a complete project, good job :fire:! 


## Evaluation criteria

| Criteria       | Indicator                                     | Yes/No |
| -------------- | --------------------------------------------- | ------ |
| 1. Is complete | Know how to answer all the above questions.   | YES    |
|                | `pandas` and `matplotlib`/`seaborn` are used. | YES    |
|                | All the above steps were followed.            | YES    |
|                | A nice README is available.                   | +/-    |
|                | Your model is able to predict something.      | YES    |
| 2. Is good     | You used typing and docstring.                | NO    |
|                | Your code is formatted (PEP8 compliant).      | NO    |
|                | No unused file/code is present.               | NO    |
