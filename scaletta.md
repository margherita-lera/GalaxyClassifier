# Scaletta progetto

- Capire in modo chiaro e univoco come spartire il lavoro


## Tasks

 1. Capire come usare le immagini con pytorch -> torch-vision??
    - Trasformazioni, vettorializzare l'immagine
 2. Capire come funziona il dataset GalaxyZoo:
    - A cosa serve il benchmark 
    - Che features sono implementate nelle soluzioni e quali realisticamente possiamo fare riconoscere al modello.
    - Ci serve una risorsa per capire anche noi quali sono le features usate per classificare
    - Come usare le training solutions.csv
 3. Setuppare la NN, capire layer maxpooling etc
 4. Provare, possibilmente in parallelo, varie NN e tracciare risultati -> Un modello a testa??
    - Pytorch tuner or whatever
 5. Fare un data augmentation anche
 6. Saliency mapping idk, vedi paper
 7. Scrivere a mano a mano quello che facciamo e capiamo in modo da facilitare la stesura del paper, tenere un [journal]{journal.md}. Anche per riferire ai colleghi cosa si è fatto
 8. Leggere letteratura a riguardo, per usare magari metodi innovativi/ poco usati ma utili per lo specifico task.

## Ruoli suggeriti da ChatGPT
| Nome  | Compiti principali                                                   |
| ----- | -------------------------------------------------------------------- |
| **A** | Dataset: studio, parsing `.csv`, trasformazioni, augmentations       |
| **B** | Architettura e training baseline + PyTorch setup                     |
| **C** | Esperimenti con modelli avanzati + hyperparameter tuning             |
| **D** | Interpretabilità (saliency maps), analisi risultati + documentazione |
| **Tutti** | Lettura paper, tenere log del proprio lavoro e scrivere update       |

