### Differential privacy

Ich glaube das war von mir nicht ganz klar formuliert gewesen, bzw. mir war das so auch zuerst nicht ganz klar:
Wenn das Verfahren differentially private ist, dann muss die Argumentation hierfür im Algorithmus liegen, nicht in der Evaluation. Sprich, wenn ein Verfahren epsilon-differential privacy gewährleistet, dann kann das Maß der Privatsphäre, das gewährleistet wird über den epsilon Parameter (das „Datenschutzbudget“) angegeben werden. "epsilon gives a ceiling on how much the probability of a particular output can increase by including (or removing) a single training example“.

Das bedeutet nicht, dass eine exakte Trajektorie eines Users sich nicht auch so im synthetischen Datensatz wiederfinden kann, aber ich kann die Wahrscheinlichkeit dessen ziemlich gut angeben. Außerdem sind die synthetischen Datensätze dann „abstreitbar“ - d.h. ein User könnte sagen, dass die Trajektorie in dem synthetischen Datensatz nur durch Rauschen entstanden ist und tatsächlich gar nicht existiert.
Also das heißt: die privacy muss nicht evaluiert werden. Was typischerweise gemacht wird, ist verschiedene Parameter für epsilon zu wählen und dann zu untersuchen, wie sich dies auf die utility auswirkt.

Etwas mehr im Detail wird das zB auch hier im Tutorial von Tensorflow beschrieben („measuring the privacy guarentee achieved"):
https://github.com/tensorflow/privacy/blob/master/tutorials/walkthrough/README.md

Hier gibts zB ein Tutorial für ein diff. Private RNN: https://github.com/tensorflow/privacy/blob/master/tutorials/lm_dpsgd_tutorial.py

Im besten Fall übernimmst du das differential privacy Modul von Tensorflow und kannst dann in der Erklärung deines Modells die bestehende Erklärung der Garantie von differential privacy von denen übernehmen.



Earth movers distance: In statistics, the earth mover's distance (EMD) is a measure of the distance between two probability distributions over a region D.

 the Kullback–Leibler divergence,(also called relative entropy), is a measure of how one probability distribution is different from a second, reference probability distribution

 the Jensen–Shannon divergence is a method of measuring the similarity between two probability distributions.

 Relative error (RE)—when used as a measure of precision—is the ratio of the absolute error of a measurement to the measurement being taken


 The 2006 Dwork, McSherry, Nissim and Smith article introduced the concept of ε-differential privacy, a mathematical definition for the privacy loss associated with any data release drawn from a statistical database. (Here, the term statistical database means a set of data that are collected under the pledge of confidentiality for the purpose of producing statistics that, by their production, do not compromise the privacy of those individuals who provided the data.)

The intuition for the 2006 definition of ε-differential privacy is that a person's privacy cannot be compromised by a statistical release if their data are not in the database. Therefore, with differential privacy, the goal is to give each individual roughly the same privacy that would result from having their data removed. That is, the statistical functions run on the database should not overly depend on the data of any one individual.

Of course, how much any individual contributes to the result of a database query depends in part on how many people's data are involved in the query. If the database contains data from a single person, that person's data contributes 100%. If the database contains data from a hundred people, each person's data contributes just 1%. The key insight of differential privacy is that as the query is made on the data of fewer and fewer people, more noise needs to be added to the query result to produce the same amount of privacy. Hence the name of the 2006 paper, "Calibrating noise to sensitivity in private data analysis."

The 2006 paper presents both a mathematical definition of differential privacy and a mechanism based on the addition of Laplace noise (i.e. noise coming from the Laplace distribution) that satisfies the definition

Epsilon (ε): A metric of privacy loss at a differentially change in data (adding, removing 1 entry). The smaller the value is, the better privacy protection.
