# Journal
Per chiarezza inserire sottoparagrafo con il vostro nome sotto il paragrafo data.

## 6/05/2025

### Gigi

Ho fatto una bozza del classificatore. Cioè la funzione che deve prendere tutte le colonne del csv e trasformarle in una classe. E' un poco difficile da leggere perchè sono alberi di if/else. Ho cercato di facilitare la lettura mettendo la response associata come commento.
To do:
 - migliorare la classificazione, magari avvicinarla a una canonica, per ora ho scritto 'smooth' e 'cigar' per intenderci.
 - Implementare il controllo suggerito da Giovanni, ovvero che se una classe non ha una ovvia risposta prominente con un certo margine, di trascurare la classe. 

## 7/05/2025
### Gigi
Allora oggi ho continuato a provare a fare un etichettatore. Gabri ne sta provando a fare uno lui anche mi ha detto.
Allora la prima cosa che ho provato a fare è stata implementare il secondo punto suggerito ieri, ovvero scegliere il maggiore di una certa soglia. Ma poi l'ho scartato perchè non avevo voglia di implementarlo, e l'opzione in cui non c'era un chiaro maggiore non sapevo come implementarla. Avrei potuto bloccare la classe allo step precendente, per cui se ero tipo SA ed ero incerto su SAb e SAc, la chiamavo SA e sti cazzi, ma non mi piace. Per cui questo punto rimane aperto.
Di contro mi sono occupato di riavvicinare la classificazione a quella di Hubble e De vaucouleurs, e direi che ci sono riuscito. Per la risposta iniziale 'smooth' sfociamo nelle E e ho scelto E0, E3 e E6. Per la risposta 'feature of disk' ho deciso che poteva essere S0, lenticolari, oppure SA/B, spirali/spirali barrate. Ho eliminato anche alcune domande, tipo quelle sul numero di bracci o sulle irregolarità, un poco per semplicità, un poco perchè anche sti cazzi, al massimo possiamo aumentare le classi in seconda sede. 
Alla fine ho ottenuto 17 classi.

## 8/05/2025
### Gigi

Con Gabri abbiamo un po' provato a capire cosa fare con le domande che hanno lo stesso numero di voti per le possibili risposte, e.g. [0.33,0.33,0.33]. Abbiamo analizzato il csv per le tre classi più scomode, che sono la Classe 9 (rounded bulge, boxy bulge, no bulge), la classe 3 (bar/no bar) e la classe 4 (spiral/not spiral). Abbiamo implementato una formulina che si chiama percentuale di somiglianza: $ 1 - ((max-min)/max)$ e abbiamo visto che le risposte difficili come (rounded/boxy) hanno un 1e5 oggetti che hanno un grado di somiglianza maggiore uguale a 0.75. Secondo noi sono troppi oggetti da scartare per cui vorremmo trovare un modo diverso di gestire la cosa. 
Noi vorremmo tenerli, per cui stiamo esplorando delle alternative.

PEr la classe 9, siccome l'indecisione è tra rounded bulge e boxy bulge, suggeriamo di fare una categoria che sia Bulge/no bulgeche si traduce in lenticolari o no.