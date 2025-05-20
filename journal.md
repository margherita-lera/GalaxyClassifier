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
Allora la prima cosa che ho provato a fare è stata implementare il secondo punto suggerito ieri, ovvero scegliere il maggiore di una certa soglia. Ma poi l'ho scartato perchè non avevo voglia di implementarlo, e l'opzione in cui non c'era un chiaro maggiore non sapevo come implementarla. Avrei potuto bloccare la classe allo step precendente, per cui se ero tipo SA ed ero incerto su SAb e SAc, la chiamavo SA, ma non mi piace. Per cui questo punto rimane aperto.
Di contro mi sono occupato di riavvicinare la classificazione a quella di Hubble e De vaucouleurs, e direi che ci sono riuscito. Per la risposta iniziale 'smooth' sfociamo nelle E e ho scelto E0, E3 e E6. Per la risposta 'feature of disk' ho deciso che poteva essere S0, lenticolari, oppure SA/B, spirali/spirali barrate. Ho eliminato anche alcune domande, tipo quelle sul numero di bracci o sulle irregolarità, un poco per semplicità, un poco perchè anche sti cazzi, al massimo possiamo aumentare le classi in seconda sede. 
Alla fine ho ottenuto 17 classi.

## 8/05/2025
### Gigi

Con Gabri abbiamo un po' provato a capire cosa fare con le domande che hanno lo stesso numero di voti per le possibili risposte, e.g. [0.33,0.33,0.33]. Abbiamo analizzato il csv per le tre classi più scomode, che sono la Classe 9 (rounded bulge, boxy bulge, no bulge), la classe 3 (bar/no bar) e la classe 4 (spiral/not spiral). Abbiamo implementato una formulina che si chiama percentuale di somiglianza: $ 1 - ((max-min)/max)$ e abbiamo visto che le risposte difficili come (rounded/boxy) hanno un 1e5 oggetti che hanno un grado di somiglianza maggiore uguale a 0.75. Secondo noi sono troppi oggetti da scartare per cui vorremmo trovare un modo diverso di gestire la cosa. 
Noi vorremmo tenerli, per cui stiamo esplorando delle alternative.

Per la classe 9, siccome l'indecisione è tra rounded bulge e boxy bulge, suggeriamo di fare una categoria che sia Bulge / no bulge che si traduce in lenticolari o no.

## 9/05/2025
### Gab e Gio

Abbiamo deciso di usare i dati senza preprocessarli. Usiamo la CNN come regressore, quindi usando le 37 colonne come labels. Quindi classificazione multi-labels (questa descrizione non è accurata). 

Reading data: abbiamo creato una classe python per leggere i dati (immagini e labels e nome). Abbiamo portato le immagini in tensori 3x426x426 e normalizzate a 1 (forse andranno ridimensionate/croppate/rinormalizzate).

Dataloader: 

## 10/05/2025
### Gigi e Marghe

Allora per quanto riguarda il dataloader, mi sembra di aver capito che basta wrappare `training` e `test`, ottenuti splittando randomicamente `DS`, con la funzione DataLoader di `pytorch`.

Oggi abbiamo abbozzato uno scheletro della CNN. Abbiamo seguito un video che la faceva con pytorch a questo ![https://www.youtube.com/watch?v=CtzfbUwrYGI]{link}.

Abbiamo quindi fatto una classe che racchude la struttura della CNN, abbiamo segnato quali sono i valori che non sono vincolati tra i vari layer nei commenti.

Abbiamo messo una loss fucntion e un optimizer ma che sono placeholder, si possono cambiare con quelle che riteniamo più efficaci.

Dopodichè abbiamo fatto un ciclo per le epoche.

Bisogna wrappare il test set.

Da capire se le label vanno bene così, la NN le comprende??. Magari può servire ricostruire la probabilità di ogni ramo dell'albero, in modo da avere classi con delle probabilità indipendenti.


## 11/05/2025
### Gab Gigi Gio

Abbiamo crosscheckato alcune cose fatte ieri, cambiato la loss function, provato a trainare, abbiamo visto che sulle GPU il tempo è 17 volte minore per il training. Abbiamo provato a usare l' eval() del modello. Ora la domanda che abbiamo è come rendere human-readable la MSELoss.

## 12/05/2025
### Tutti

Abbiamo parlato di croppare le immagini (non necessario se è veloce, da vedere). Basta inserirlo nella trasformazione iniziale selezionando i pixel che ci interessano.
Abbiamo parlato di non usare i tre canali rgb ma usare le immagini in bianco e nero. Questo è da vedere se mantiene la stessa performance e confrontarla con quella a tre canali. 
Stiamo facendo una regressione: da capire come stabilire se le nostre performances sono buone, anche se usare il MSE come loss va bene. (meglio RMSE)

### Marghe

Inserito il crop e la possibilità di mettere in bianco e nero le immagini nel training. Automatizzato la scelta di parametri nella CNN, ora basta cambiare all'inizio le variabili locali.
Ho anche pulito il main di git. Se vi serve NaturalClassifier o altro scrivetemeli che li ho in locale e ve li mando.

## 13/05/2025

### Gab Gigi Gio

Gio ha rotto il suo pc.

Il dottorando ci ha consigliato un analogo di `keras tuner` per pytorch che si chiama optuna. Ci ha dato un occhio Gab nella lezione di oggi.
Pensiamo che potrebbe essere la scelta migliore per tunare gli iperparametri.

Gigi dovrebbe aver sistemato la parte di salvataggio e ri-caricamento dei modelli e dei vari parametri.

## 14/05/2025

### Gigi

Stavo guardando la transforms.compose. Ho aggiunto random Horizontal/vertical flip. E anche ho pensato che se vogliamo fare sudare un po la CNN possiamo aggiungere un randomCrop prima del più consistente CenterCrop in modo da decentrare un poco le immagini.

Ho rifatto il ciclo di training e evaluation mettendolo in due funzioni, solo perchè è + bello e ordinato in realtà. Ora sto modificando la classe `GalaxyNet` in modo tale da salvare tutte le metriche possibili. Per ora sto salvando:
 - `mini-batch loss`
 - `epoch mean loss`
 - `validation batch loss`
 - `validation epoch mean loss`

Poi pensavo di salvare cose tipo numero epoche/batch sizes e tutte ste cose qui.  Ho aggiunto al salvataggio del modello tutti i parametri che mi sono venuti in mente. 

## 16/05/2025

### Gigi e Gab

Ieri Gab è riuscito a far funzionare uno study di optuna. Poi con Gigi hanno deciso di rifare tutto il setup di optuna da capo, per comprendere passaggio dopo passaggio cosa stesse succedendo. Hanno creato una classe per la CNN e impostato uno scheletro che poi la CNN potrà tunare in molti aspetti. Su suggerimento di giovanni si sono informati sulla BatchNorm e sembra che possa evitare di usare dropout e possa permettere di velocizzare il processo di apprendimento della NN in quanto permette Lr + alti. Si sono informati sulla inizializzazione di bias e pesi. Con batch norm sembra che l'inizializzazione del bias non serva in quanto non viene usato per l'aggiornamento dei pesi. Per l'inizializzazione dei pesi invece hanno scelto l'inizializzazione di He o di Kaiming che sembra essere fatta apposta per le attivazioni Relu e Leaky Relu.
Manca da fare il forward, e altre cose.

### Marghe
Ho creato la funzione mapper. Prende in ingresso il tensore di label e ritorna un tensore con le 17 classi create da gigi. Le label corrispondono a probabilità indipendenti che ho ricavato 'a mano'.
Per vedere l'output sotto forma di dataframe con i nomi delle label associati basta mettere datafr=True in ingresso.
Ho messo la funzione da sola in un mappy.py così è più facile da inputtare nel nostro notebook.
Ho anche scoperto che a livello di performance è molto meglio lavorare direttamente con il tensore in 2D piuttosto che creare la funzione per ogni riga e poi mapparla su tutte le righe con altre funzioni o con iterazione. Forse era scontato ma nel dubbio lascio qui l'info per i posteri.

Resta da capire se è compatibile con la nostra neural network, in particolare le label finali ovviamente dovranno cambiare nomi e ci dobbiamo mettere l'argmax per avere una risposta finale.
Secondo me sarebbe interessante fare qualche distribuzione di probabilità, magari divisa in macrocategorie sommando tutte le ellittiche e spirali per farci un'idea pratica di come è fatta questa popolazione.
### Marghe, Gigi

Marghe e gigi alla fine hanno troubleshootato la struttura della CNN, e hanno implementato una funzione per il train. L'ultimo layer è attivato da una sigmoide invece che da `F.linear`, perchè quest'ultima voleva dei pesi in argomento.
Gab invece ha definito una funzione `objective` che ha mandato nel gruppo.

## 17/05/2025

### Gigi

Gigi ha unito e sistemato quello che ha fatto Gabri ieri, con quello fatto da Marghe e Gigi ieri. Ha aggiunto una funzione `validationxepoca`
### Gio

Fixed rgb handling inside the NN class.
In gio branch.



Fixed rgb handling inside the NN class. 

Minor * aesthetic * changes in the convolutional cycle.

Put batchnorm before pooling. It might go before ReLU even, but it is probably something to test out. 

Got rid of sigmoid until we mappily use mappy. 

Fixed the evaluation error(?). It used to compute the mean by the last RMSE value, now it uses the running one. 

Put nn.ReLU() instead of F.relu inside initialization of NN. 

Forgot I could write in Italian in here.


### Gigi

In gigi branch.

Gigi ha unito e sistemato quello che ha fatto Gabri ieri, con quello fatto da Marghe e Gigi ieri. Ha aggiunto una funzione `validationxepoca`

In gio branch.

Fixed rgb handling inside the NN class.
Minor * aesthetic * changes in the convolutional cycle. 
Put batchnorm before pooling. It might go before ReLU even, but it is probably something to test out. Got rid of sigmoid until we mappily use mappy.
Fixed the evaluation error(?). It used to compute the mean by the last RMSE value, now it uses the running one.
Put nn.ReLU() instead of F.relu inside initialization of NN.
Forgot I could write in Italian in here.
## 19/05/2025

### Gio

Sistemato optuna. Da considerare quando si runna: batch size, epoche, trials.

Forse si può ottimizzare meglio per ridurre il tempo di run. Togliere log the loss per ogni singola batch loss?


## 20/05/2025

### Tutti (a parte Margherita, ma è solo un personaggio secondario)

Ottimizzato il tempo di run riducendo la "precisione" dei float.

Sistemato esteticamente la definizione della rete e la funzione forward.

Sistemato init\_weights.

Bisogna decidere le architetture da comparare, non ha probabilmente senso usare CNN profonde, ma è una cosa che si può controllare. La nostra rete con 3 strati convolutivi esplode perché l'output si riduce ad ogni passaggio, ha senso usare il padding per non far ridurre l'output? Senza padding è inutile il ciclo per definire gli strati convolutivi, se lasciamo stare la questione profondità esigo che venga sterminato il ciclo e sia tutto ben definito nella classe nn.Sequential.

Il professore ha suggerito di scrivere tutte le considerazioni che abbiamo fatto finora sul paper.
