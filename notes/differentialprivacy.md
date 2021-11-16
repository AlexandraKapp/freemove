### Differential privacy

Ich glaube das war von mir nicht ganz klar formuliert gewesen, bzw. mir war das so auch zuerst nicht ganz klar:
Wenn das Verfahren differentially private ist, dann muss die Argumentation hierfür im Algorithmus liegen, nicht in der Evaluation. Sprich, wenn ein Verfahren epsilon-differential privacy gewährleistet, dann kann das Maß der Privatsphäre, das gewährleistet wird über den epsilon Parameter (das „Datenschutzbudget“) angegeben werden. "epsilon gives a ceiling on how much the probability of a particular output can increase by including (or removing) a single training example“.

Das bedeutet nicht, dass eine exakte Trajektorie eines Users sich nicht auch so im synthetischen Datensatz wiederfinden kann, aber ich kann die Wahrscheinlichkeit dessen ziemlich gut angeben. Außerdem sind die synthetischen Datensätze dann „abstreitbar“ - d.h. ein User könnte sagen, dass die Trajektorie in dem synthetischen Datensatz nur durch Rauschen entstanden ist und tatsächlich gar nicht existiert.
Also das heißt: die privacy muss nicht evaluiert werden. Was typischerweise gemacht wird, ist verschiedene Parameter für epsilon zu wählen und dann zu untersuchen, wie sich dies auf die utility auswirkt.

Etwas mehr im Detail wird das zB auch hier im Tutorial von Tensorflow beschrieben („measuring the privacy guarentee achieved"):
https://github.com/tensorflow/privacy/blob/master/tutorials/walkthrough/README.md

Hier gibts zB ein Tutorial für ein diff. Private RNN: https://github.com/tensorflow/privacy/blob/master/tutorials/lm_dpsgd_tutorial.py

Im besten Fall übernimmst du das differential privacy Modul von Tensorflow und kannst dann in der Erklärung deines Modells die bestehende Erklärung der Garantie von differential privacy von denen übernehmen.
