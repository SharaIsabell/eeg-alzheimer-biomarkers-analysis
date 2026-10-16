# Resumo dos resultados (gerado por models_training.ipynb)

Sujeitos: 65. Classe majoritária: 55.4%. IC95%: Wilson (acurácia) e bootstrap estratificado com 2000 réplicas (AUC).

| Modelo | Conjunto | Acurácia | AUC | Sensib. | Especif. |
|---|---|---|---|---|---|
| SVM (Linear) | Todas (31) | 78.5% [67.0%; 86.7%] | 0.880 [0.787; 0.944] | 75.0% | 82.8% |
| XGBoost | Todas (31) | 72.3% [60.4%; 81.7%] | 0.872 [0.775; 0.942] | 72.2% | 72.4% |
| Random Forest | Todas (31) | 76.9% [65.4%; 85.5%] | 0.862 [0.759; 0.938] | 75.0% | 79.3% |
| SVM (Linear) | Kuramoto (sensor) | 80.0% [68.7%; 87.9%] | 0.859 [0.757; 0.939] | 77.8% | 82.8% |
| SVM (Linear) | RBP | 76.9% [65.4%; 85.5%] | 0.853 [0.744; 0.940] | 63.9% | 93.1% |
| Random Forest | LZC (banda estreita) | 80.0% [68.7%; 87.9%] | 0.846 [0.736; 0.941] | 83.3% | 75.9% |
| Random Forest | RBP | 76.9% [65.4%; 85.5%] | 0.840 [0.735; 0.930] | 77.8% | 75.9% |
| SVM (Linear) | Informação mútua | 73.8% [62.0%; 83.0%] | 0.836 [0.723; 0.926] | 69.4% | 79.3% |
| Random Forest | Informação mútua | 72.3% [60.4%; 81.7%] | 0.828 [0.717; 0.919] | 77.8% | 65.5% |
| XGBoost | LZC (banda estreita) | 76.9% [65.4%; 85.5%] | 0.824 [0.709; 0.922] | 83.3% | 69.0% |
| XGBoost | Informação mútua | 72.3% [60.4%; 81.7%] | 0.820 [0.704; 0.916] | 77.8% | 65.5% |
| XGBoost | RBP | 72.3% [60.4%; 81.7%] | 0.819 [0.705; 0.915] | 72.2% | 72.4% |
| SVM (Linear) | LZC (banda estreita) | 75.4% [63.7%; 84.2%] | 0.814 [0.693; 0.912] | 72.2% | 79.3% |
| SVM (Linear) | Centroide espectral | 67.7% [55.6%; 77.8%] | 0.796 [0.685; 0.893] | 63.9% | 72.4% |
| Random Forest | Kuramoto (sensor) | 80.0% [68.7%; 87.9%] | 0.795 [0.670; 0.903] | 77.8% | 82.8% |
| XGBoost | Kuramoto (sensor) | 76.9% [65.4%; 85.5%] | 0.779 [0.650; 0.892] | 80.6% | 72.4% |
| Random Forest | Centroide espectral | 73.8% [62.0%; 83.0%] | 0.762 [0.640; 0.868] | 75.0% | 72.4% |
| XGBoost | Centroide espectral | 64.6% [52.5%; 75.1%] | 0.708 [0.575; 0.829] | 66.7% | 62.1% |
| XGBoost | PAC local | 50.8% [38.9%; 62.5%] | 0.510 [0.366; 0.651] | 55.6% | 44.8% |
| SVM (Linear) | PAC local | 50.8% [38.9%; 62.5%] | 0.494 [0.350; 0.639] | 63.9% | 34.5% |
| Random Forest | PAC local | 50.8% [38.9%; 62.5%] | 0.486 [0.343; 0.635] | 61.1% | 37.9% |

## Comparações (McNemar exato)

| Modelo | Tipo | A | B | Acc A | Acc B | só A acerta | só B acerta | p | p Holm |
|---|---|---|---|---|---|---|---|---|---|
| Random Forest | primária | Todas (31) | RBP | 0.769 | 0.769 | 3 | 3 | 1.0000 | 1.0000 |
| Random Forest | secundária (exploratória) | Todas (31) | Kuramoto (sensor) | 0.769 | 0.800 | 4 | 6 | 0.7539 |  |
| SVM (Linear) | primária | Todas (31) | RBP | 0.785 | 0.769 | 7 | 6 | 1.0000 | 1.0000 |
| SVM (Linear) | secundária (exploratória) | Todas (31) | Kuramoto (sensor) | 0.785 | 0.800 | 5 | 6 | 1.0000 |  |
| XGBoost | primária | Todas (31) | RBP | 0.723 | 0.723 | 8 | 8 | 1.0000 | 1.0000 |
| XGBoost | secundária (exploratória) | Todas (31) | Kuramoto (sensor) | 0.723 | 0.769 | 6 | 9 | 0.6072 |  |

Modelo só com idade e sexo: AUC 0.580 [0.442; 0.725].
