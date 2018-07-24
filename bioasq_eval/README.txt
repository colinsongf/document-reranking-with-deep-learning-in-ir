This software is part of the [BioASQ Evaluation-Measures repository](https://github.com/BioASQ/Evaluation-Measures).


Evaluation Measures for BioASQ Challenge
-----------------------------------------------

Instructions for BioASQ evaluation measures
-----------------------------------------------

Task B
---------

1. For running the measures for Task B, phase A the following command is invoked:

java -Xmx10G -cp $CLASSPATH:./flat/BioASQEvaluation/dist/BioASQEvaluation.jar evaluation.EvaluatorTask1b -phaseA -e 5 golden_file.json system_response.json
