# EVO2-Virus

**A  EVO2 architecture model trained on over 40,000 viruses CDNA sequence.**

## 1. **Architectural Advantages**

* ​**Highly customizable model** based on the modified Vortex inference engine
* ​**Flexible GPU layer allocation** (user-defined layer-to-GPU mapping)

## 2. Datasets

| Species | NUM |
|:-------|:-------:|
| **Eukaryote-infecting viruses** | 13221  | 
| **Prokaryote-infecting viruses** | 27130  |

## 3. Checkpoints

| Checkpoint Name |Description |
|:-------|:-------:|
| **EVO2_Virus_500M** | A model finetuned base on EVO-1B checkpoint  | 
| **Coming Soon** |Larger models trained on more data will be released in the future |

## 4.Perfermence

**We tested 500 untrained eukaryote-infecting viruses and 500 prokaryote-infecting viruses for evaluation.**

| Model |Similarity |Score|
|:-------|:-------:|:-------:|
| **EVO2_Virus_500M** | 64.12% | 281.2 |
| **EVO2_1b** |0% | -52 |
| **EVO2_7b** |0% | -52 |
| **EVO2_40b** |0% | -52 |

## 5.**Availability**

**"Checkpoints, training code, and framework code will be provided upon request to xuzhixiang210@gmail.com through you. All content is completely free, but we require assurance that you will not use it to generate harmful material!"**


